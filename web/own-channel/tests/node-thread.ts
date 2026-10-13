import {parentPort} from 'node:worker_threads';
import type {CPUThreadRequest, CPUThreadResponse} from '../dist/cpu-protocol.js';
if (!parentPort) throw new Error('CPU thread must run in a worker.');
const port = parentPort;
const workerGlobal = globalThis as unknown as {postMessage(message: CPUThreadResponse): void; onmessage: ((event: MessageEvent<CPUThreadRequest>) => void) | null};
workerGlobal.postMessage = message => port.postMessage(message);
workerGlobal.onmessage = null;
await import('../dist/cpu-thread.js');
port.on('message', (data: CPUThreadRequest) => workerGlobal.onmessage?.(new MessageEvent('message', {data})));
