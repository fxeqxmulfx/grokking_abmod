export const LANGUAGES = Object.freeze({en: 'English', es: 'Español', de: 'Deutsch', ja: '日本語', fr: 'Français', ru: 'Русский'});
type Locale = keyof typeof LANGUAGES;
type Catalog = Record<string, string>;
interface Binding {node: Node; source: string; attribute?: string; prefix?: string; suffix?: string}
interface Template {pattern: RegExp; names: string[]; translated: string}
const isLocale = (value: string | null): value is Locale => value !== null && Object.hasOwn(LANGUAGES, value);
const STORAGE_KEY = 'grokking.locale', catalogs = new Map<Locale, Promise<Catalog>>(), bindings: Binding[] = [];
let locale: Locale = 'en', selection = 'auto', messages: Catalog = {}, runtimeTemplates: Template[] = [], generation = 0, captured = false;

export function chooseLocale(preferred: readonly string[] = [], explicit = 'auto'): Locale {
  if (isLocale(explicit)) return explicit;
  for (const tag of preferred) {
    const language = String(tag).toLowerCase().split(/[-_]/)[0];
    if (isLocale(language)) return language;
  }
  return 'en';
}
export const currentLocale = () => locale;
export function t(message: string, values: Record<string, string | number> = {}) {
  const translated = Object.hasOwn(messages, message) ? messages[message] : message;
  return translated.replace(/\{(\w+)\}/g, (match, name) => Object.hasOwn(values, name) ? String(values[name]) : match);
}
const formats = new Map<string, Intl.NumberFormat>();
export function number(value: number, options: Intl.NumberFormatOptions = {}) {
  const key = locale + JSON.stringify(options);
  if (!formats.has(key)) formats.set(key, new Intl.NumberFormat(locale, options));
  return formats.get(key)!.format(value);
}
export function runtimeText(message: string): string {
  if (!message) return '';
  if (Object.hasOwn(messages, message)) return t(message);
  // Backend statuses stay stable English identifiers. Translate known runtime
  // messages at the UI boundary, including values in HTTP/error templates.
  for (const {pattern, names, translated} of runtimeTemplates) {
    const match = message.match(pattern);
    if (match) return translated.replace(/\{(\w+)\}/g, (_: string, key: string) => {
      const value = match[names.indexOf(key) + 1];
      return key === 'error' || key === 'node' ? runtimeText(value) : Object.hasOwn(messages, value) ? messages[value] : value;
    });
  }
  return message;
}

function applyStatic() {
  for (const {node, source, attribute, prefix = '', suffix = ''} of bindings) {
    if (!node.isConnected) continue;
    const value = prefix + t(source) + suffix;
    if (attribute && node instanceof Element) node.setAttribute(attribute, value);
    else node.nodeValue = value;
  }
}
function captureStatic() {
  if (captured) return; captured = true;
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const node = walker.currentNode;
    if (node.parentElement?.closest('script, style, pre, code, var, canvas')) continue;
    const source = (node.nodeValue ?? "").trim();
    if (source) bindings.push({node, source, prefix: (node.nodeValue ?? "").match(/^\s*/)![0], suffix: (node.nodeValue ?? "").match(/\s*$/)![0]});
  }
  for (const node of document.querySelectorAll('[aria-label], [title], [alt], [placeholder], meta[name="description"]')) {
    for (const attribute of ['aria-label', 'title', 'alt', 'placeholder', 'content']) if (node.hasAttribute(attribute)) bindings.push({node, source: node.getAttribute(attribute)!, attribute});
  }
}
function refreshSelectors() {
  for (const select of document.querySelectorAll<HTMLSelectElement>('[data-language-select]')) {
    if (!select.options.length) select.replaceChildren(new Option('', 'auto'), ...Object.entries(LANGUAGES).map(([code, name]) => new Option(name, code)));
    select.options[0].textContent = `${t('Auto')} · ${LANGUAGES[locale]}`;
    select.value = selection;
    select.title = t('Language'); select.setAttribute('aria-label', t('Language'));
  }
}
export async function setLanguage(value: string, {persist = true} = {}) {
  const request = ++generation;
  selection = Object.hasOwn(LANGUAGES, value) ? value : 'auto';
  const selected = chooseLocale(navigator.languages?.length ? navigator.languages : [navigator.language], selection);
  if (!catalogs.has(selected)) catalogs.set(selected, import(`./locales/${selected}.js`).then(module => module.default));
  const dictionary = await catalogs.get(selected)!;
  if (request !== generation) return;
  locale = selected; messages = dictionary;
  runtimeTemplates = Object.entries(messages).filter(([source]) => source.includes('{')).map(([source, translated]) => {
    const names: string[] = [], parts = source.split(/(\{\w+\})/).map(part => {
      if (/^\{\w+\}$/.test(part)) { names.push(part.slice(1, -1)); return '(.*?)'; }
      return part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    });
    return {pattern: new RegExp('^' + parts.join('') + '$', 's'), names, translated};
  });
  if (persist) { try { localStorage.setItem(STORAGE_KEY, selection); } catch { /* Private storage can be unavailable. */ } }
  document.documentElement.lang = locale;
  applyStatic(); refreshSelectors();
  window.dispatchEvent(new CustomEvent('localechange', {detail: {locale, selection}}));
}
export async function initializeI18n() {
  captureStatic();
  let preferred = 'auto';
  try { preferred = localStorage.getItem(STORAGE_KEY) ?? 'auto'; } catch { /* Keep automatic selection. */ }
  const override = new URL(location.href).searchParams.get('lang');
  if (isLocale(override)) preferred = override;
  await setLanguage(preferred, {persist: false});
  for (const select of document.querySelectorAll<HTMLSelectElement>('[data-language-select]')) select.onchange = () => setLanguage(select.value);
  window.addEventListener('languagechange', () => { if (selection === 'auto') setLanguage('auto', {persist: false}); });
}
