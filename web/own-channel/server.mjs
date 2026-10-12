import {createServer} from 'node:http';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

const root = path.dirname(fileURLToPath(import.meta.url));
const portIndex = process.argv.indexOf('--port');
const port = portIndex < 0 ? 8080 : Number(process.argv[portIndex + 1]);
const mime = {'.html': 'text/html; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.wgsl': 'text/plain; charset=utf-8', '.wasm': 'application/wasm'};
const server = createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    const relative = pathname.startsWith('/tests/') || pathname.startsWith('/dist/') ? pathname.slice(1) : 'dist/' + (pathname === '/' ? 'index.html' : pathname.slice(1));
    const filename = path.resolve(root, relative);
    if (!filename.startsWith(root + path.sep) || !['dist', 'tests'].includes(path.relative(root, filename).split(path.sep)[0]) || !mime[path.extname(filename)]) {
      response.writeHead(404); response.end('Not found'); return;
    }
    const content = await readFile(filename);
    response.writeHead(200, {'Content-Type': mime[path.extname(filename)], 'Cache-Control': 'no-store'}); response.end(content);
  } catch { response.writeHead(404); response.end('Not found'); }
});
server.listen(port, '127.0.0.1', () => console.log(`Own-channel Grokking Lab: http://127.0.0.1:${server.address().port}`));
