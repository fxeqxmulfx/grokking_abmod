import {Worker} from 'node:worker_threads';
import type {CPUThread, CPUThreadResponse} from '../dist/cpu-protocol.js';
export function nodeCPUThread(): CPUThread {
  const worker = new Worker(new URL('./node-thread.js', import.meta.url));
  const port: CPUThread = {postMessage: message => worker.postMessage(message), terminate: () => {void worker.terminate();}, onmessage: null, onerror: null};
  worker.on('message', (data: CPUThreadResponse) => port.onmessage?.({data}));
  worker.on('error', (error: Error) => port.onerror?.({message: error.message}));
  worker.on('exit', code => {if (code !== 0) port.onerror?.({message: `CPU worker exited with code ${code}.`});});
  return port;
}
