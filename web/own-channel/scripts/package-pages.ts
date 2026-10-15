import {cp, mkdir, readdir, rm, stat, writeFile} from 'node:fs/promises';
import {root} from '../project.ts';

const source = new URL('dist/', root), destination = new URL('build/pages/', root);
await rm(destination, {recursive: true, force: true});
await mkdir(destination, {recursive: true});
let files = 0, bytes = 0;
async function copy(relative = '') {
  for (const entry of await readdir(new URL(relative, source), {withFileTypes: true})) {
    const name = relative + entry.name;
    if (entry.isDirectory()) {await mkdir(new URL(name + '/', destination), {recursive: true}); await copy(name + '/');}
    else if (entry.isFile() && /\.(html|css|js|wasm|svg|txt)$|\.js\.map$/.test(entry.name)) {
      await cp(new URL(name, source), new URL(name, destination));
      files++; bytes += (await stat(new URL(name, destination))).size;
    }
  }
}
await copy();
await writeFile(new URL('.nojekyll', destination), '');
console.log(`GitHub Pages: ${files} files, ${(bytes / 1048576).toFixed(2)} MiB in build/pages/.`);
