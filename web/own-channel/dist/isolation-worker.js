const isolation = globalThis;
isolation.addEventListener('install', event => event.waitUntil(isolation.skipWaiting()));
isolation.addEventListener('activate', event => event.waitUntil(isolation.clients.claim()));
isolation.addEventListener('fetch', event => {
    if (new URL(event.request.url).origin !== isolation.location.origin)
        return;
    if (event.request.cache === 'only-if-cached' && event.request.mode !== 'same-origin')
        return;
    // Fetch from the host on every request; deployments do not leave an old
    // asset cache behind. Only same-origin responses receive these headers.
    event.respondWith(fetch(event.request).then(response => {
        if (response.status === 0)
            return response;
        const headers = new Headers(response.headers);
        headers.set('Cross-Origin-Opener-Policy', 'same-origin');
        headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
        headers.set('Cross-Origin-Resource-Policy', 'same-origin');
        return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
    }));
});
export {};
//# sourceMappingURL=isolation-worker.js.map