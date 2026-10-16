import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

// All generated PDF files stay in the repository's ignored target directory.
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const reportDirectory = join(root, 'docs/grokking-results/div97-mlp');
const buildDirectory = join(root, 'target/div97-report');
const figureDirectory = join(buildDirectory, 'figures');
const markdown = join(reportDirectory, 'report.md');
const pdf = join(buildDirectory, 'div97-mlp-report.pdf');
for (const directory of [reportDirectory, buildDirectory, figureDirectory]) {
  mkdirSync(directory, { recursive: true });
}
if (!existsSync(markdown)) throw new Error(`Report source is missing: ${markdown}`);

const palette = {
  ink: '#142638', muted: '#526476', line: '#526a80', border: '#cbd5df',
  blue: '#2563a6', blueFill: '#edf4fc', green: '#147967', greenFill: '#edf8f4',
  purple: '#7153a2', purpleFill: '#f4effb', orange: '#a75b1e', orangeFill: '#fff5e9',
};
const escape = (value) => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const text = (x, y, content, size = 20, fill = palette.ink, anchor = 'middle', weight = 400) =>
  `<text x="${x}" y="${y}" font-size="${size}" fill="${fill}" text-anchor="${anchor}" font-weight="${weight}">${escape(content)}</text>`;
const line = (x1, y1, x2, y2, color = palette.line) =>
  `<path d="M ${x1} ${y1} L ${x2} ${y2}" fill="none" stroke="${color}" stroke-width="2.5" marker-end="url(#${color === palette.green ? 'skip' : 'arrow'})"/>`;
const path = (d, color = palette.green) =>
  `<path d="${d}" fill="none" stroke="${color}" stroke-width="2.5" stroke-linejoin="round" marker-end="url(#${color === palette.green ? 'skip' : 'arrow'})"/>`;
function box(x, y, width, height, lines, { fill = palette.blueFill, stroke = palette.blue, size = 21 } = {}) {
  const spacing = size + 8;
  return `<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="12" fill="${fill}" stroke="${stroke}" stroke-width="1.6"/>` +
    lines.map((label, index) => text(x + width / 2, y + height / 2 + 7 + (index - (lines.length - 1) / 2) * spacing, label, size, palette.ink, 'middle', index === 0 ? 600 : 400)).join('');
}
function addition(x, y) {
  return `<circle cx="${x}" cy="${y}" r="19" fill="${palette.greenFill}" stroke="${palette.green}" stroke-width="2"/>` + text(x, y + 8, '+', 29, palette.green);
}
function svg(width, height, title, description, content) {
  return `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title description">
<title id="title">${escape(title)}</title><desc id="description">${escape(description)}</desc>
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="${palette.line}"/></marker>
  <marker id="skip" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="${palette.green}"/></marker>
</defs>
<rect width="${width}" height="${height}" fill="white"/>
<g font-family="DejaVu Sans, sans-serif">${content}</g>
</svg>\n`;
}

const parameterCount = (width, depth) => (depth - 1) * width ** 2 + (depth + 3) * width + 1;
function overview() {
  let content = text(38, 47, 'Two scaled inputs → one scalar prediction', 31, palette.ink, 'start', 700);
  content += text(38, 84, 'D counts all ReLU hidden layers, including the first projection. W is hidden width.', 20, palette.muted, 'start');
  content += box(38, 146, 145, 104, ['Inputs', '(aₛ, bₛ)']);
  content += box(218, 146, 202, 104, ['Affine 2 → W', 'ReLU']);
  content += box(455, 146, 250, 104, ['W → W block', 'repeat D − 1 times', 'choose A–D'], { size: 20, fill: palette.purpleFill, stroke: palette.purple });
  content += box(740, 146, 205, 104, ['Affine W → 1', 'linear output']);
  content += box(980, 146, 182, 104, ['Inverse scale', 'round to integer'], { size: 20, fill: palette.greenFill, stroke: palette.green });
  for (const [start, end] of [[183, 212], [420, 449], [705, 734], [945, 974]]) content += line(start, 198, end, 198);
  content += text(1071, 281, 'accuracy only', 19, palette.green);
  content += box(455, 319, 250, 83, ['Target cₛ', 'same numeric scale'], { fill: '#f6f8fa', stroke: palette.border, size: 20 });
  content += box(740, 319, 205, 83, ['MSE loss', 'before rounding'], { fill: palette.orangeFill, stroke: palette.orange, size: 20 });
  content += line(705, 360, 734, 360) + line(842, 250, 842, 313);
  const variants = [[128, 4, 'matched depth control'], [128, 8, 'optimization reference'], [256, 8, 'best training recipe'], [128, 16, 'deep-grid family']];
  for (let index = 0; index < variants.length; index++) {
    const [width, depth, label] = variants[index];
    const x = 38 + 287 * index;
    content += `<rect x="${x}" y="453" width="263" height="122" rx="12" fill="#f7f9fc" stroke="${palette.border}"/>`;
    content += text(x + 131.5, 486, `W${width} × ${depth}`, 25, palette.ink, 'middle', 600);
    content += text(x + 131.5, 520, `${parameterCount(width, depth).toLocaleString('en-US')} parameters`, 20, palette.muted);
    content += text(x + 131.5, 551, label, 18, palette.muted);
  }
  content += text(38, 617, 'Trainable biases in every affine layer. No normalization or skip on the input projection or readout.', 19, palette.muted, 'start');
  return svg(1200, 645, 'Scalar MLP overview', 'Input projection, repeated width-preserving blocks, a linear scalar readout, an unrounded MSE loss branch, and rounded accuracy evaluation. Representative tested widths and depths are shown with parameter counts.', content);
}

function hiddenVariants() {
  let content = text(28, 46, 'Four hidden-block variants', 31, palette.ink, 'start', 700);
  content += text(28, 84, 'Each block preserves W features; each repeated layer has its own affine weights and bias.', 20, palette.muted, 'start');
  const variants = [
    { label: 'A · Plain ReLU', norm: false, skip: false, formula: 'h_next = ReLU(A h + b)' },
    { label: 'B · Pre-RMSNorm', norm: true, skip: false, formula: 'h_next = ReLU(A RMSNorm(h) + b)' },
    { label: 'C · Residual', norm: false, skip: true, formula: 'h_next = h + ReLU(A h + b)' },
    { label: 'D · Pre-RMSNorm + residual', norm: true, skip: true, formula: 'h_next = h + ReLU(A RMSNorm(h) + b)' },
  ];
  for (let index = 0; index < variants.length; index++) {
    const variant = variants[index];
    const x = index % 2 ? 614 : 28;
    const y = index < 2 ? 118 : 507;
    const cy = y + 181;
    content += `<rect x="${x}" y="${y}" width="558" height="358" rx="15" fill="#fbfcfe" stroke="${palette.border}" stroke-width="1.5"/>`;
    content += text(x + 24, y + 39, variant.label, 23, palette.ink, 'start', 600);
    content += text(x + 24, y + 71, `normalization ${variant.norm ? 'on' : 'off'} · skip ${variant.skip ? 'on' : 'off'}`, 19, palette.muted, 'start');
    content += text(x + 25, cy + 7, 'h', 22);
    if (variant.norm) {
      content += box(x + 62, cy - 37, 115, 74, ['RMSNorm'], { size: 19, fill: palette.purpleFill, stroke: palette.purple });
      content += box(x + 202, cy - 37, 130, 74, ['Affine', 'W → W'], { size: 20 });
      content += box(x + 357, cy - 37, 88, 74, ['ReLU'], { size: 19 });
      content += line(x + 39, cy, x + 56, cy) + line(x + 183, cy, x + 196, cy) + line(x + 338, cy, x + 351, cy);
    } else {
      content += box(x + 101, cy - 37, 208, 74, ['Affine W → W', 'with bias'], { size: 21 });
      content += box(x + 344, cy - 37, 101, 74, ['ReLU'], { size: 20 });
      content += line(x + 39, cy, x + 95, cy) + line(x + 315, cy, x + 338, cy);
    }
    if (variant.skip) {
      content += line(x + 451, cy, x + 462, cy) + addition(x + 487, cy) + line(x + 512, cy, x + 522, cy);
      content += path(`M ${x + 45} ${cy} L ${x + 45} ${y + 111} L ${x + 487} ${y + 111} L ${x + 487} ${cy - 25}`);
      content += text(x + 271, y + 103, 'identity', 18, palette.green);
    } else {
      content += line(x + 451, cy, x + 522, cy);
    }
    content += text(x + 537, cy + 7, 'h′', 21);
    content += text(x + 24, y + 281, variant.formula, 18, palette.ink, 'start');
    content += text(x + 24, y + 321, variant.skip ? 'Skip carries h before any normalization.' : 'Only the branch becomes the next hidden state.', 18, palette.muted, 'start');
  }
  content += text(28, 911, 'RMSNorm: epsilon 10⁻⁵; no learned gain or bias. D = 1 has no eligible block and uses one common baseline.', 19, palette.muted, 'start');
  return svg(1200, 940, 'Scalar MLP hidden-block variants', 'Four panels show plain ReLU, pre-RMSNorm, residual ReLU, and pre-RMSNorm with residual. Residual arrows carry the original hidden state around the complete branch and merge after ReLU.', content);
}

function normalization() {
  let content = text(28, 46, 'Input and target normalization', 31, palette.ink, 'start', 700);
  content += text(28, 83, 'Same fixed transform for both operands and the target. The scalar readout remains linear.', 20, palette.muted, 'start');
  const modes = [
    { name: 'Raw', inputs: '(a, b)', target: 'c', restored: 'ŷ' },
    { name: 'Unit · divide by 96', inputs: '(a/96, b/96)', target: 'c/96', restored: '96 ŷ' },
    { name: 'Centered · divide by 96, subtract 0.5', inputs: '(a/96 − 0.5, b/96 − 0.5)', target: 'c/96 − 0.5', restored: '96 (ŷ + 0.5)' },
  ];
  for (const [index, mode] of modes.entries()) {
    const y = 111 + index * 155;
    content += `<rect x="28" y="${y}" width="1144" height="139" rx="14" fill="#fbfcfe" stroke="${palette.border}"/>`;
    content += text(48, y + 29, mode.name, 23, palette.ink, 'start', 600);
    content += box(48, y + 47, 430, 75, [`Inputs: ${mode.inputs}`, `Target: ${mode.target}`], { size: 20 });
    content += box(523, y + 47, 205, 75, ['MLP output ŷ', 'unbounded'], { size: 21, fill: palette.purpleFill, stroke: palette.purple });
    content += box(775, y + 47, 370, 75, [`Restore: ${mode.restored}`, 'then round for accuracy'], { size: 21, fill: palette.greenFill, stroke: palette.green });
    content += line(484, y + 84.5, 517, y + 84.5) + line(734, y + 84.5, 769, y + 84.5);
  }
  content += text(28, 602, 'MSE uses the unrounded output ŷ and the transformed target, before inverse scaling.', 20, palette.muted, 'start');
  return svg(1200, 630, 'MLP input and target normalization', 'Three rows show raw, unit-scaled and centered operands and training targets. Unit scaling divides both inputs and the target by 96; centering also subtracts 0.5. The inverse output transform restores residue units before rounding for accuracy.', content);
}

const figures = [
  ['scalar-mlp-overview.svg', overview()],
  ['hidden-block-variants.svg', hiddenVariants()],
  ['input-target-normalization.svg', normalization()],
];
for (const [name, source] of figures) writeFileSync(join(reportDirectory, name), source);

const environment = {
  ...process.env,
  XDG_CONFIG_HOME: join(buildDirectory, 'config'),
  XDG_CACHE_HOME: join(buildDirectory, 'cache'),
  DIV97_REPORT_FIGURES: figureDirectory,
};
mkdirSync(environment.XDG_CONFIG_HOME, { recursive: true });
mkdirSync(environment.XDG_CACHE_HOME, { recursive: true });
let buildLog = '';
function run(command, args) {
  const result = spawnSync(command, args, { cwd: root, env: environment, encoding: 'utf8', maxBuffer: 32 * 1024 * 1024 });
  buildLog += `${command} ${args.join(' ')}\n${result.stdout ?? ''}${result.stderr ?? ''}\n`;
  writeFileSync(join(buildDirectory, 'build.log'), buildLog);
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} failed with status ${result.status}; see ${join(buildDirectory, 'build.log')}`);
  return `${result.stdout ?? ''}${result.stderr ?? ''}`.trim();
}
const versions = Object.fromEntries(['pandoc', 'xelatex', 'inkscape'].map(command => {
  const output = run(command, ['--version']);
  const firstLine = output.split('\n').find(value => value.toLowerCase().includes(command));
  return [command, firstLine ?? output.split('\n')[0]];
}));
for (const [name] of figures) {
  run('inkscape', [join(reportDirectory, name), '--export-area-page', '--export-type=pdf', `--export-filename=${join(figureDirectory, name.replace(/\.svg$/, '.pdf'))}`]);
}

const filter = join(buildDirectory, 'figures.lua');
writeFileSync(filter, String.raw`function Image(image)
  if image.src:match('%.svg$') then
    local filename = image.src:match('([^/]+)%.svg$')
    image.src = os.getenv('DIV97_REPORT_FIGURES') .. '/' .. filename .. '.pdf'
  end
  return image
end

function RawBlock(block)
  if block.format == 'html' and block.text:match('^<!%-%- pagebreak %-%->$') then
    return pandoc.RawBlock('latex', '\\clearpage')
  end
end
`);
const header = join(buildDirectory, 'report-header.tex');
writeFileSync(header, String.raw`\usepackage{float}
\floatplacement{figure}{H}
\usepackage{fancyhdr}
\usepackage{caption}
\usepackage{titlesec}
\definecolor{reportblue}{HTML}{255A83}
\definecolor{reportmuted}{HTML}{526476}
\titleformat{\section}{\sffamily\bfseries\LARGE\color{reportblue}}{}{0pt}{}
\titleformat{\subsection}{\sffamily\bfseries\large\color{reportblue}}{}{0pt}{}
\captionsetup{font={small,sf},labelfont=bf,skip=5pt}
\pagestyle{fancy}
\fancyhf{}
\fancyfoot[L]{\sffamily\footnotesize\color{reportmuted}Division modulo 97 · Local Rust/CUDA study}
\fancyfoot[R]{\sffamily\footnotesize\color{reportmuted}\thepage}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0.3pt}
\AtBeginDocument{\hypersetup{pdftitle={Division modulo 97: local MLP investigation},pdfsubject={Scalar ReLU MLP architectures and input and target normalization},pdfauthor={Local experiment report}}}
\setlength{\emergencystretch}{3em}
\setlength{\tabcolsep}{5pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
`);
run('pandoc', [markdown, '--standalone', '--from=markdown', '--pdf-engine=xelatex', `--lua-filter=${filter}`, `--include-in-header=${header}`, `--resource-path=${reportDirectory}`, '--variable=documentclass:article', '--variable=fontsize:11pt', '--variable=geometry:a4paper,top=19mm,bottom=21mm,left=19mm,right=19mm', '--variable=mainfont:TeX Gyre Pagella', '--variable=sansfont:TeX Gyre Heros', '--variable=monofont:DejaVu Sans Mono', '--variable=colorlinks:true', '--variable=linkcolor:reportblue', '--variable=urlcolor:reportblue', '--variable=linestretch:1.04', '--output', pdf]);

const sha256 = file => createHash('sha256').update(readFileSync(file)).digest('hex');
const sourceEvidence = join(root, 'target/local-ablations-2026-10-16/mlp-investigation.json');
const manifest = {
  generatedAt: new Date().toISOString(),
  scope: 'Division modulo 97, scalar MLP only. Documentation export; no new training.',
  pdfGitPolicy: 'PDF and intermediate outputs are under the already ignored target directory.',
  versions,
  sources: Object.fromEntries([markdown, fileURLToPath(import.meta.url), ...figures.map(([name]) => join(reportDirectory, name))].map(file => [file.slice(root.length + 1), sha256(file)])),
  evidence: existsSync(sourceEvidence) ? { path: sourceEvidence.slice(root.length + 1), sha256: sha256(sourceEvidence) } : null,
  output: { path: pdf.slice(root.length + 1), sha256: sha256(pdf), bytes: readFileSync(pdf).length },
};
writeFileSync(join(buildDirectory, 'build-manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
console.log(`Wrote ${figures.length} editable architecture SVGs to ${reportDirectory.slice(root.length + 1)}/`);
console.log(`Compiled ${pdf.slice(root.length + 1)} (${manifest.output.bytes.toLocaleString('en-US')} bytes)`);
