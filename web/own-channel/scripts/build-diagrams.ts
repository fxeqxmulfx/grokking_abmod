import {CDP} from '../tests/cdp.ts';
import {root} from '../project.ts';
import {readFile, writeFile, mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {languages, rows} from './translations.ts';

// Render the real Mermaid source once during development. The site only
// loads the small resulting SVG; no Mermaid runtime or CDN is needed.
const port = Number(process.env.GROKKING_CDP_PORT || 9224);
const client = await CDP.connect(port);
const call = client.call.bind(client), evaluate = client.evaluate.bind(client), errors = client.errors;
const waitFor = client.wait.bind(client), wait = waitFor;
let layouts = 0;
const metadata: {renderer: string; version: string; diagrams: Record<string, {sourceSha256: string; svgSha256: string; bytes: number}>} = {renderer: '@mermaid-js/tiny', version: '12.1.0', diagrams: {}};
try {
  await call('Page.navigate', {url: 'about:blank'});
  await evaluate('new Promise(resolve=>setTimeout(resolve,100))');
  const library = await readFile(new URL('node_modules/@mermaid-js/tiny/dist/mermaid.tiny.js', root), 'utf8');
  await evaluate(library + '\nmermaid.initialize(' + JSON.stringify({startOnLoad:false, securityLevel:'strict', htmlLabels:false, layout:'dagre', theme:'base', deterministicIds:true,
    fontFamily:'Arial, sans-serif', themeVariables:{fontSize:'16px', primaryColor:'#f1f1e9', primaryTextColor:'#252d28', primaryBorderColor:'#d4d8cd', lineColor:'#49849c'},
    markdownAutoWrap:false, flowchart:{htmlLabels:false, useMaxWidth:false, wrappingWidth:300, nodeSpacing:18, rankSpacing:18, padding:10, curve:'linear'}}) + '); true;');
  await mkdir(new URL('dist/diagrams/', root), {recursive:true});
  await mkdir(new URL('diagrams/', root), {recursive:true});
  for (const [index, language] of languages.entries()) {
    const dictionary = Object.fromEntries(rows.map(row => [row[0], row[index]])), translate = (source: string) => dictionary[source] ?? source;
    const definition = [
      'flowchart TB',
      `accTitle: ${translate('Mathematical solution')}`,
      `accDescr: ${translate('For 10 / 3, the inverse of 3 is 65. The exact answer is 68, and multiplying it by 3 restores remainder 10.')}`,
      `A["${translate('Inputs')}<br/>a = 10, b = 3"]`,
      `B["${translate('Find the inverse')}<br/>3 · 65 = 2 · 97 + 1"]`,
      `C["${translate('Multiply and take the remainder')}<br/>10 · 65 = 6 · 97 + 68"]`,
      `D["${translate('Verify the answer')}<br/>3 · 68 = 2 · 97 + 10"]`,
      `Q["${translate('Answer')}: q = 68"]`,
      'A --> B --> C --> D --> Q',
      'style Q fill:#e1efde,stroke:#2d8975,stroke-width:2px',
      '',
    ].join('\n');
    const svg = await evaluate<string>(`mermaid.render(${JSON.stringify('division-'+language)}, ${JSON.stringify(definition)}).then(result=>result.svg)`);
    if (!svg.includes('<svg') || svg.includes('<script') || svg.includes('<foreignObject')) throw new Error(`Unsafe or invalid diagram: ${language}`);
    const localized = svg.replace('<svg ', `<svg lang="${language}" `);
    await writeFile(new URL(`diagrams/division-${language}.mmd`, root), definition);
    await writeFile(new URL(`dist/diagrams/division-${language}.svg`, root), localized + '\n');
    metadata.diagrams[language] = {sourceSha256:createHash('sha256').update(definition).digest('hex'), svgSha256:createHash('sha256').update(localized+'\n').digest('hex'), bytes:Buffer.byteLength(localized+'\n')};
    console.log(`${language}: ${metadata.diagrams[language].bytes} SVG bytes`);
  }
  await writeFile(new URL('diagrams/manifest.json', root), JSON.stringify(metadata,null,2)+'\n');
} finally { client.close(); }
