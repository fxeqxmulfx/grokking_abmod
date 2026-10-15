import {element as $} from './dom.ts';
import {t} from './i18n.ts';
import {HELP_TOPICS} from './help-content.ts';
import type {GuidePage, HelpId, HelpTopic} from './help-content.ts';

export function initializeHelp() {
  const dialog = $('help-dialog');
  const buttons = document.querySelectorAll<HTMLButtonElement>('button[data-help]');
  let current: HelpId | null = null;
  const topic = (): HelpTopic => HELP_TOPICS[current!];
  function render() {
    if (!current) return;
    $('help-title').textContent = t(topic().title);
    const paragraphs = topic().paragraphs.map(source => {
      const paragraph = document.createElement('p'); paragraph.textContent = t(source); return paragraph;
    });
    if (topic().formula) {
      const paragraph = document.createElement('p'), code = document.createElement('code');
      code.textContent = topic().formula!; paragraph.append(code); paragraphs.push(paragraph);
    }
    $('help-content').replaceChildren(...paragraphs);
    $('help-guide').hidden = !topic().guide;
    $('help-content').scrollTop = 0;
  }
  function localizeButtons() {
    for (const button of buttons) {
      const id = button.dataset.help as HelpId;
      if (!Object.hasOwn(HELP_TOPICS, id)) throw new Error(`Unknown help topic: ${id}`);
      const label = t('Help: {topic}', {topic: t(HELP_TOPICS[id].title)});
      button.setAttribute('aria-label', label); button.title = label;
    }
  }
  for (const button of buttons) button.onclick = event => {
    event.preventDefault(); event.stopPropagation();
    current = button.dataset.help as HelpId; render();
    if (!dialog.open) dialog.showModal();
  };
  for (const id of ['help-close', 'help-done'] as const) $(id).onclick = () => dialog.close();
  dialog.onclick = event => {
    const rect = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom)) dialog.close();
  };
  $('help-guide').onclick = () => {
    const guide: GuidePage | undefined = topic().guide;
    if (!guide) return;
    dialog.close(); $('task-help').click();
    if (guide === 'task' || guide === 'model') $(`task-tab-${guide}`).click();
    else $(`intro-tab-${guide}`).click();
  };
  window.addEventListener('localechange', () => {localizeButtons(); render();});
  localizeButtons();
}
