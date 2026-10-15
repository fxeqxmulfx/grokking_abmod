import {root} from '../project.ts';
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {chooseLocale, LANGUAGES, t} from '../dist/i18n.js';
import {languages, rows} from '../scripts/translations.ts';

test('automatic language selection respects browser order and regional tags', () => {
  assert.equal(chooseLocale(['es-MX', 'en-US']), 'es');
  assert.equal(chooseLocale(['de-AT']), 'de');
  assert.equal(chooseLocale(['ja-JP']), 'ja');
  assert.equal(chooseLocale(['fr-CA']), 'fr');
  assert.equal(chooseLocale(['ru-KZ', 'en']), 'ru');
  assert.equal(chooseLocale(['zh-CN', 'uk-UA', 'fr-FR']), 'fr');
  assert.equal(chooseLocale(['zh-CN']), 'en');
  assert.equal(chooseLocale([]), 'en');
});
test('manual choice overrides browser language, invalid choices fall back safely', () => {
  assert.equal(chooseLocale(['ja-JP'], 'ru'), 'ru');
  assert.equal(chooseLocale(['ru-RU'], 'en'), 'en');
  assert.equal(chooseLocale(['fr-FR'], '../../unexpected'), 'fr');
  assert.equal(chooseLocale(['fr-FR'], 'constructor'), 'fr');
  assert.equal(t('constructor'), 'constructor');
  assert.deepEqual(Object.keys(LANGUAGES), languages);
});
test('all catalogs contain every message and preserve interpolation fields', async () => {
  const placeholders = (value: string) => Array.from(value.matchAll(/\{(\w+)\}/g), match => match[1]).sort();
  for (const [index, language] of languages.entries()) {
    const dictionary = (await import(new URL(`dist/locales/${language}.js`, root).href)).default;
    assert.equal(Object.keys(dictionary).length, rows.length);
    for (const row of rows) {
      assert.equal(dictionary[row[0]], row[index]);
      assert.deepEqual(placeholders(row[index]), placeholders(row[0]));
    }
  }
});
test('all initial accessible labels and titles have translations', async () => {
  const html = await readFile(new URL('dist/index.html', root), 'utf8');
  const source = (await import('../dist/locales/en.ts')).default;
  for (const match of html.matchAll(/(?:aria-label|title|alt|placeholder)="([^"]+)"/g)) assert.ok(Object.hasOwn(source, match[1]), match[1]);
  assert.ok(!html.includes('EOS') && !html.includes('causal mask'));
  assert.equal((html.match(/data-task-tab=/g) ?? []).length, 5);
  assert.equal((html.match(/data-intro-tab=/g) ?? []).length, 7);
});
test('every division diagram is a rendered local Mermaid SVG matching its source manifest', async () => {
  const manifest = JSON.parse(await readFile(new URL('diagrams/manifest.json', root), 'utf8'));
  assert.equal(manifest.renderer, '@mermaid-js/tiny');
  for (const language of languages) {
    const source = await readFile(new URL(`diagrams/division-${language}.mmd`, root), 'utf8');
    const svg = await readFile(new URL(`dist/diagrams/division-${language}.svg`, root), 'utf8');
    assert.ok(source.startsWith('flowchart TB'));
    assert.ok(svg.includes('<svg') && svg.includes(`lang="${language}"`));
    assert.ok(!svg.includes('<script') && !svg.includes('<foreignObject'));
    assert.ok(svg.replace(/<[^>]*>/g, '').replace(/\s+/g, '').includes('q=68'));
    assert.equal(createHash('sha256').update(source).digest('hex'), manifest.diagrams[language].sourceSha256);
    assert.equal(createHash('sha256').update(svg).digest('hex'), manifest.diagrams[language].svgSha256);
  }
});
