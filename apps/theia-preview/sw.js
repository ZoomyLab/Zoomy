/* Zoomy theia-preview service worker — offline (#10) + cross-origin isolation.
 *
 * 1. COOP/COEP injection: GitHub Pages doesn't send these, so SharedArrayBuffer
 *    (the kernel's cooperative interrupt / Stop) is unavailable. We add them to
 *    every response so the page becomes `crossOriginIsolated`.
 * 2. Offline: cache the app shell + gui/ assets + the CDN libs + PyPI wheels, so
 *    after the first (online) visit the whole GUI — kernel included — works with
 *    no network.
 */
var CACHE = 'zoomy-preview-v1';
// Cache these origins/paths (same-origin app + gui/, Pyodide/CDN libs, wheels).
var CACHEABLE = /(^\/|jsdelivr\.net|esm\.sh|files\.pythonhosted\.org|pypi\.org|cdn\.jsdelivr)/;

self.addEventListener('install', function (e) { self.skipWaiting(); });
self.addEventListener('activate', function (e) {
    e.waitUntil((async function () {
        var keys = await caches.keys();
        await Promise.all(keys.filter(function (k) { return k !== CACHE; }).map(function (k) { return caches.delete(k); }));
        await self.clients.claim();
    })());
});
self.addEventListener('message', function (e) {
    if (e.data && e.data.type === 'deregister') { self.registration.unregister(); }
});

function withCoi(resp) {
    if (!resp || resp.status === 0) { return resp; }
    var h = new Headers(resp.headers);
    h.set('Cross-Origin-Embedder-Policy', 'require-corp');
    h.set('Cross-Origin-Opener-Policy', 'same-origin');
    h.set('Cross-Origin-Resource-Policy', 'cross-origin');
    return new Response(resp.body, { status: resp.status, statusText: resp.statusText, headers: h });
}

self.addEventListener('fetch', function (e) {
    var req = e.request;
    if (req.method !== 'GET') { return; }
    var url = new URL(req.url);
    var sameOrigin = url.origin === self.location.origin;
    var cacheable = sameOrigin || CACHEABLE.test(url.host) || CACHEABLE.test(url.href);
    e.respondWith((async function () {
        var cache = await caches.open(CACHE);
        var cached = await cache.match(req);
        // Cache-first (offline + speed) for cacheable GETs; always revalidate in bg.
        var network = fetch(req).then(function (resp) {
            try { if (cacheable && resp && resp.status === 200 && resp.type !== 'opaque') { cache.put(req, resp.clone()); } } catch (x) { }
            return withCoi(resp);
        }).catch(function () { return cached ? withCoi(cached.clone()) : Response.error(); });
        if (cached) {
            // Serve cached immediately (offline-capable); refresh cache in background.
            network.catch(function () { });
            return withCoi(cached.clone());
        }
        return network;
    })());
});
