export function browserCPUThread() {
    const worker = new Worker(new URL('./cpu-thread.js', import.meta.url), { type: 'module' });
    const port = { postMessage: message => worker.postMessage(message), terminate: () => worker.terminate(), onmessage: null, onerror: null };
    worker.onmessage = (event) => port.onmessage?.({ data: event.data });
    worker.onerror = event => port.onerror?.({ message: event.message });
    return port;
}
//# sourceMappingURL=cpu-protocol.js.map