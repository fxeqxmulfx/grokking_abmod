import {createServer} from 'node:http';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {root as projectRoot} from './project.ts';
import {modelForWidth} from './src/models.ts';
import {CPU_CONFIG} from './scripts/cpu-config.ts';

const root = fileURLToPath(projectRoot), portIndex = process.argv.indexOf('--port');
const port = portIndex < 0 ? 8080 : Number(process.argv[portIndex + 1]);
if (!Number.isInteger(port) || port < 0 || port > 65535) throw new Error('Invalid preview port.');
const mime: Record<string, string> = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.wasm': 'application/wasm', '.svg': 'image/svg+xml', '.map': 'application/json'};
// Compile missing widths on demand and retain only a small set of modules.
// Training and weights remain in the browser; this serves generated assets.
const customModules = new Map<string, Promise<Uint8Array<ArrayBuffer>>>();
async function customModule(name: string, width: number, variant: 'simd' | 'scalar', shared: boolean) {
  const model = modelForWidth(width);
  if (!customModules.has(name)) {
    if (customModules.size >= 16) customModules.delete(customModules.keys().next().value!);
    const compilation = import('./scripts/assemble-cpu.ts').then(({assembleCPU}) => assembleCPU(model, variant, CPU_CONFIG[`${width >= 256 ? 256 : 128}-${variant}`], shared));
    customModules.set(name, compilation);
    compilation.catch(() => {if (customModules.get(name) === compilation) customModules.delete(name);});
  }
  return customModules.get(name)!;
}
const server = createServer(async (request, response) => {
  // Shared WASM memory requires an isolated document; every asset is local.
  response.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  response.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  try {
    const pathname = decodeURIComponent(new URL(request.url ?? '/', 'http://localhost').pathname);
    const relative = pathname.startsWith('/tests/')
      ? (pathname.endsWith('.js') ? 'build' : '') + pathname
      : pathname.startsWith('/dist/') ? pathname.slice(1) : 'dist/' + (pathname === '/' ? 'index.html' : pathname.slice(1));
    const filename = path.resolve(root, relative.replace(/^\//, ''));
    const allowed = ['dist', 'tests', 'build'].includes(path.relative(root, filename).split(path.sep)[0]);
    if (!filename.startsWith(root) || !allowed || !mime[path.extname(filename)]) {
      response.writeHead(404); response.end('Not found'); return;
    }
    let content: Uint8Array;
    try {content = await readFile(filename);}
    catch (error) {
      const custom = /^dense-rotary-([1-9]\d*)-(simd|scalar)(-shared)?\.wasm$/.exec(path.basename(filename));
      if (!custom || relative !== 'dist/' + custom[0] || (error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
      content = await customModule(custom[0], Number(custom[1]), custom[2] as 'simd' | 'scalar', Boolean(custom[3]));
    }
    response.writeHead(200, {'Content-Type': mime[path.extname(filename)], 'Cache-Control': 'no-store'}); response.end(content);
  } catch {response.writeHead(404); response.end('Not found');}
});
server.listen(port, '127.0.0.1', () => {
  const address = server.address();
  if (address && typeof address !== 'string') console.log(`Rotate90 Grokking Lab: http://127.0.0.1:${address.port}`);
});
