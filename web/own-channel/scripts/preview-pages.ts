import {createServer} from 'node:http';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {root} from '../project.ts';

// A plain static host under a project prefix, without isolation headers.
// This exercises the same URLs and service-worker bootstrap as GitHub Pages.
const directory = fileURLToPath(new URL('build/pages/', root)), base = '/grokking_abmod/';
const portIndex = process.argv.indexOf('--port'), port = portIndex < 0 ? 8081 : Number(process.argv[portIndex + 1]);
if (!Number.isInteger(port) || port < 0 || port > 65535) throw new Error('Invalid preview port.');
const mime: Record<string, string> = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.wasm': 'application/wasm', '.svg': 'image/svg+xml', '.map': 'application/json', '.txt': 'text/plain; charset=utf-8'};
createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url ?? '/', 'http://localhost').pathname);
    if (pathname === base.slice(0, -1)) {response.writeHead(301, {Location: base}); response.end(); return;}
    if (!pathname.startsWith(base)) throw new Error('Not found');
    const filename = path.resolve(directory, pathname.slice(base.length) || 'index.html');
    const relative = path.relative(directory, filename);
    if (relative.startsWith('..') || path.isAbsolute(relative) || !mime[path.extname(filename)]) throw new Error('Not found');
    const content = await readFile(filename);
    response.writeHead(200, {'Content-Type': mime[path.extname(filename)], 'Cache-Control': 'no-store'}); response.end(content);
  } catch {response.writeHead(404); response.end('Not found');}
}).listen(port, '127.0.0.1', () => console.log(`GitHub Pages preview: http://127.0.0.1:${port}${base}`));
