import type {CPUAddresses} from './cpu-memory.ts';

export type CPUThreadRequest =
  | {type: 'initialize'; module: WebAssembly.Module; memory: WebAssembly.Memory; addresses: CPUAddresses; control: number; slot: number}
  | {type: 'gradient'; job: number; source: number; first: number; count: number; scale: number}
  | {type: 'evaluate'; job: number; first: number; end: number};
export type CPUThreadResponse = {type: 'ready' | 'done'; job: number} | {type: 'error'; message: string};
export interface CPUThread {
  postMessage(message: CPUThreadRequest): void;
  terminate(): void;
  onmessage: ((event: {data: CPUThreadResponse}) => void) | null;
  onerror: ((event: {message: string}) => void) | null;
}
export function browserCPUThread(): CPUThread {
  const worker = new Worker(new URL('./cpu-thread.js', import.meta.url), {type: 'module'});
  const port: CPUThread = {postMessage: message => worker.postMessage(message), terminate: () => worker.terminate(), onmessage: null, onerror: null};
  worker.onmessage = (event: MessageEvent<CPUThreadResponse>) => port.onmessage?.({data: event.data});
  worker.onerror = event => port.onerror?.({message: event.message});
  return port;
}
