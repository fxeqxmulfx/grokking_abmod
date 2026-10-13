import {root} from '../project.ts';
import {mkdir, writeFile} from 'node:fs/promises';
import {languages, rows} from './translations.ts';

// Native SVG keeps the architecture legible in both viewport orientations.
// Dashed connections indicate weight reuse; arrows carry activations.
const directory = new URL('dist/diagrams/', root);
await mkdir(directory, {recursive: true});
const escape = (value: string) => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const titleSize = 18, formulaSize = 16;

function titleLines(value: string, width: number): string[] {
  const units = (line: string) => Array.from(line).reduce((sum, character) => sum + (/[\u2e80-\ua4cf\uac00-\ud7af\uff01-\uff60]/u.test(character) ? 1 : .62), 0);
  const lines: string[] = [];
  for (const word of value.split(' ')) {
    const previous = lines.at(-1);
    if (previous && units(`${previous} ${word}`) * titleSize <= width) lines[lines.length - 1] = `${previous} ${word}`;
    else lines.push(word);
  }
  return lines;
}

function diagram(language: string, translate: (message: string) => string, portrait: boolean) {
  const width = portrait ? 400 : 1020, height = portrait ? 642 : 350;
  const elements: string[] = [];
  function text(x: number, y: number, value: string, style = '') {
    return `<text x="${x}" y="${y}" ${style}>${escape(value)}</text>`;
  }
  function box(id: string, x: number, y: number, w: number, h: number, title: string, formulas: string[] = [], tone = 'fixed', titleClass = 'label') {
    // Wrap translated titles at a fixed size; mathematical symbols never shrink.
    const titles = titleClass === 'formula' ? [title] : titleLines(title, w - 16);
    const titleHeight = (titleClass === 'formula' ? formulaSize : titleSize) + 2, formulaHeight = formulaSize + 2;
    const total = titles.length * titleHeight + (formulas.length ? 2 + formulas.length * formulaHeight : 0);
    if (total > h - 6) throw new Error(`${language}/${id}: increase the box height to fit its labels.`);
    let top = y + (h - total) / 2, content = '';
    for (const line of titles) {
      content += text(x + w / 2, top + titleHeight / 2, line, `text-anchor="middle" dominant-baseline="central" class="${titleClass}"`);
      top += titleHeight;
    }
    if (formulas.length) top += 2;
    for (const formula of formulas) {
      content += text(x + w / 2, top + formulaHeight / 2, formula, 'text-anchor="middle" dominant-baseline="central" class="formula"');
      top += formulaHeight;
    }
    elements.push(`<g id="${id}" class="${tone}"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10"/>${content}</g>`);
  }
  function add(id: string, x: number, y: number, name: string) {
    const label = portrait ? text(x + 29, y + (id === 'input-sum' ? 14 : 0), name, 'dominant-baseline="central" class="formula"') : text(id === 'input-sum' ? x - 38 : x, y + 43, name, 'text-anchor="middle" class="formula"');
    const plus = `<path class="operator" d="M${x - 5} ${y}H${x + 5}M${x} ${y - 5}V${y + 5}"/>`;
    elements.push(`<g id="${id}" class="fixed"><circle cx="${x}" cy="${y}" r="18"/>${plus}${label}</g>`);
  }
  function path(id: string, d: string, style = 'flow') { elements.push(`<path id="${id}" d="${d}" class="${style}"/>`); }

  if (portrait) {
    elements.push('<rect class="branch" x="64" y="224" width="278" height="190" rx="14"/>');
    path('shared-a', 'M100 60V72', 'shared');
    path('shared-b', 'M300 60V72', 'shared');
    path('shared-readout', 'M360 34H390V542H335', 'shared');
    path('a-input', 'M99 114V196H182');
    path('b-input', 'M301 114V126');
    path('rotated-input', 'M301 176V196H218');
    path('normalize-input', 'M200 214V230');
    path('dense-input', 'M200 270V277H251V284');
    path('dense-weights', 'M156 319H174', 'parameter');
    path('relu-input', 'M251 354V358H200V362');
    path('branch-output', 'M200 402V412');
    path('residual-skip', 'M182 196H42V430H182', 'skip');
    path('normalize-output', 'M200 448V462');
    path('readout-input', 'M200 502V516');
    path('prediction', 'M200 568V584');
    box('embedding', 40, 8, 320, 52, translate('Shared E'), ['97 × W'], 'learned');
    box('lookup-a', 24, 72, 150, 42, 'a → E[a]', [], 'learned', 'formula');
    box('lookup-b', 226, 72, 150, 42, 'b → E[b]', [], 'learned', 'formula');
    box('rotation', 228, 126, 146, 50, 'Rotate90', ['(u,v) → (−v,u)']);
    add('input-sum', 200, 196, 'x₀');
    box('input-norm', 136, 230, 128, 40, 'RMSNorm');
    box('dense-matrix', 74, 284, 82, 70, 'A', ['W × W'], 'learned', 'formula');
    box('dense-multiply', 174, 284, 154, 70, 'MatMul', ['z = r · A']);
    box('relu', 102, 362, 196, 40, translate('ReLU'));
    add('residual-sum', 200, 430, 'x₁');
    box('output-norm', 136, 462, 128, 40, 'RMSNorm');
    box('readout', 65, 516, 270, 52, translate('Tied readout'), ['logits = h · Eᵀ'], 'learned');
    box('answer', 65, 584, 270, 50, translate('97 logits'), ['q̂ = argmax(logits)'], 'answer');
    elements.push(text(35, 320, 'x₀', 'text-anchor="end" class="muted formula"'));
    elements.push(text(282, 257, 'r', 'class="formula"'));
    elements.push(text(282, 489, 'h', 'class="formula"'));
  } else {
    elements.push('<rect class="branch" x="278" y="112" width="554" height="136" rx="14"/>');
    elements.push(text(294, 139, `1 × ${translate('Residual block')}`, 'class="muted annotation"'));
    path('shared-a', 'M104 84V166', 'shared');
    path('shared-b', 'M104 84V116H18V295H40', 'shared');
    path('shared-readout', 'M235 52H1014V299H1008', 'shared');
    path('a-input', 'M168 193H214');
    path('b-input', 'M168 295H194');
    path('rotated-input', 'M250 268V252H232V211');
    path('normalize-input', 'M250 193H290');
    path('dense-input', 'M396 193H424');
    path('dense-weights', 'M528 264V228', 'parameter');
    path('relu-input', 'M632 193H660');
    path('branch-output', 'M764 193H782');
    path('residual-skip', 'M232 175V100H800V175', 'skip');
    path('normalize-output', 'M818 193H854');
    path('readout-input', 'M917 220V264');
    path('prediction', 'M826 299H802');
    box('embedding', 30, 20, 205, 64, translate('Shared E'), ['97 × W'], 'learned');
    box('lookup-a', 40, 166, 128, 54, 'a → E[a]', [], 'learned', 'formula');
    box('lookup-b', 40, 268, 128, 54, 'b → E[b]', [], 'learned', 'formula');
    box('rotation', 194, 268, 112, 54, 'Rotate90', ['(−v,u)']);
    add('input-sum', 232, 193, 'x₀');
    box('input-norm', 290, 166, 106, 54, 'RMSNorm', ['r ∈ ℝᵂ']);
    box('dense-matrix', 454, 264, 148, 70, 'A', ['A ∈ ℝᵂˣᵂ'], 'learned', 'formula');
    box('dense-multiply', 424, 158, 208, 70, 'MatMul', ['z = r · A']);
    box('relu', 660, 166, 104, 54, translate('ReLU'), ['max(0,z)']);
    add('residual-sum', 800, 193, 'x₁');
    box('output-norm', 854, 166, 126, 54, 'RMSNorm', ['h ∈ ℝᵂ']);
    box('readout', 826, 264, 182, 70, translate('Tied readout'), ['logits = h · Eᵀ'], 'learned');
    box('answer', 608, 264, 194, 70, translate('97 logits'), ['q̂ = argmax(logits)'], 'answer');
    elements.push(text(514, 90, `${translate('Skip connection')} · x₀`, 'text-anchor="middle" class="muted annotation"'));
    elements.push(text(408, 185, 'r', 'text-anchor="middle" class="formula"'));
    elements.push(text(646, 185, 'z', 'text-anchor="middle" class="formula"'));
  }
  const title = translate('Rotate90 model architecture');
  const description = `${translate('Rotate90 predicts one of 97 answer classes q from (a,b). One shared E embeds both operands and reads out the answer.')} x₀ = E[a] + Rotate90(E[b]); r = RMSNorm(x₀); MatMul: z = r · A, (1 × W) · (W × W) → 1 × W; ∀c ∈ {0,…,W−1}: z[c] = ∑ᵢ r[i]A[i,c]; x₁ = x₀ + ReLU(z); logits = RMSNorm(x₁) · Eᵀ; q̂ = min argmax(logits).`;
  return `<!-- Generated by scripts/build-model-diagrams.ts. -->
<svg xmlns="http://www.w3.org/2000/svg" lang="${language}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title description">
<title id="title">${escape(title)}</title><desc id="description">${escape(description)}</desc>
<defs>
  <marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L8 4L0 8Z" fill="#49849c"/></marker>
  <marker id="skip-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#7d8178"/></marker>
  <marker id="parameter-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#cf8a88"/></marker>
</defs>
<style>
  text { font-family: Arial, 'Noto Sans CJK JP', sans-serif; font-size: ${titleSize}px; fill: #252d28; }
  .label { font-weight: 600; }
  .formula { font-family: 'DejaVu Sans Mono', monospace; font-size: ${formulaSize}px; font-weight: 400; }
  .annotation { font-size: ${formulaSize}px; }
  .operator { fill: none; stroke: #252d28; stroke-width: 1.6; stroke-linecap: round; }
  .muted { fill: #7d8178; }
  .fixed rect, .fixed circle { fill: #fffefa; stroke: #d4d8cd; stroke-width: 1.5; }
  .learned rect { fill: #f8e9e8; stroke: #cf8a88; stroke-width: 1.5; }
  .answer rect { fill: #e1efde; stroke: #2d8975; stroke-width: 1.5; }
  .branch { fill: #f1f1e9; stroke: #e4e5da; }
  .flow { fill: none; stroke: #49849c; stroke-width: 2; marker-end: url(#arrow); stroke-linejoin: round; }
  .skip { fill: none; stroke: #7d8178; stroke-width: 2; marker-end: url(#skip-arrow); stroke-linejoin: round; }
  .parameter { fill: none; stroke: #cf8a88; stroke-width: 1.8; marker-end: url(#parameter-arrow); stroke-linejoin: round; }
  .shared { fill: none; stroke: #cf8a88; stroke-width: 1.8; stroke-dasharray: 5 5; stroke-linejoin: round; }
</style>
${elements.join('\n')}
</svg>
`;
}

for (const [index, language] of languages.entries()) {
  const dictionary = Object.fromEntries(rows.map(row => [row[0], row[index]]));
  const translate = (source: string) => dictionary[source] ?? source;
  for (const portrait of [false, true]) {
    const svg = diagram(language, translate, portrait);
    const filename = `model-${language}${portrait ? '-portrait' : ''}.svg`;
    await writeFile(new URL(filename, directory), svg);
    console.log(`${filename}: ${Buffer.byteLength(svg)} SVG bytes`);
  }
}
