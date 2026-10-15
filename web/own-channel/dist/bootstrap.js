// Static hosts cannot set isolation headers. A scoped worker supplies them
// before the app starts, keeping shared CPU training available on Pages.
const isolationKey = `grokking.isolation:${new URL('./', import.meta.url).pathname}`;
if (globalThis.crossOriginIsolated) {
    try {
        sessionStorage.removeItem(isolationKey);
    }
    catch { /* Storage is optional. */ }
}
else if (globalThis.isSecureContext && 'serviceWorker' in navigator) {
    try {
        const controlled = new Promise(resolve => navigator.serviceWorker.addEventListener('controllerchange', () => resolve(), { once: true }));
        const worker = new URL('./isolation-worker.js', import.meta.url);
        await navigator.serviceWorker.register(worker, { type: 'module', scope: './' });
        if (!navigator.serviceWorker.controller)
            await Promise.race([controlled, new Promise(resolve => setTimeout(resolve, 2500))]);
        if (navigator.serviceWorker.controller?.scriptURL === worker.href && !sessionStorage.getItem(isolationKey)) {
            sessionStorage.setItem(isolationKey, '1');
            location.reload();
            await new Promise(() => { });
        }
    }
    catch { /* One CPU thread remains available without service workers. */ }
}
await import('./app.js');
export {};
//# sourceMappingURL=bootstrap.js.map