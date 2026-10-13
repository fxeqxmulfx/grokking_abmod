import {cpSync, readdirSync, rmSync, existsSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {root} from '../project.ts';

function run(args: string[]): void {
  const result = spawnSync(process.execPath, args, {cwd: fileURLToPath(root), stdio: 'inherit'});
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}
// Node 24 runs this small bootstrap and the catalog generator. All runtime,
// server, test and build-tool TypeScript is compiled with strict checking.
// TypeScript does not remove outputs of deleted sources. Retire only known
// obsolete GPU assets, leaving historical experiment checkpoints intact.
for (const directory of ['dist/', 'build/dist/', 'build/src/', 'build/tests/']) {
  const location = new URL(directory, root);
  if (!existsSync(location)) continue;
  for (const filename of readdirSync(location)) if (filename.endsWith('.wgsl') || /^(gpu|models-gpu|measure-gpu|measure-dense-gpu)\.(js|d\.ts|js\.map)$/.test(filename)) rmSync(new URL(filename, location));
}
run(['scripts/build-locales.ts']);
run(['node_modules/typescript/bin/tsc', '-p', 'tsconfig.json']);
run(['node_modules/typescript/bin/tsc', '-p', 'tsconfig.tools.json']);
run(['build/scripts/build-wasm.js']);
run(['build/scripts/build-raster.js']);
run(['build/scripts/build-model-diagrams.js']);

// Node validation uses the same emitted browser modules and binary assets.
cpSync(new URL('dist/', root), new URL('build/dist/', root), {recursive: true});
