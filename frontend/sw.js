/* RoadGuard Service Worker (sw.js) */
const CACHE_NAME = "roadguard-cache-v5";
const ASSETS_TO_CACHE = [
  "/",
  "/style.css",
  "/style_v3.css",
  "/script.js",
  "/features_v3.js",
  "/gallery.html",
  "/manifest.json",
  "/icon-192.png",
  "/icon-512.png"
];

self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      return cache.addAll(ASSETS_TO_CACHE);
    })
  );
  self.skipWaiting();
});

self.addEventListener("activate", event => {
  event.waitUntil(
    caches.keys().then(keys => {
      return Promise.all(
        keys.map(key => {
          if (key !== CACHE_NAME) {
            return caches.delete(key);
          }
        })
      );
    })
  );
  self.clients.claim();
});

self.addEventListener("fetch", event => {
  if (event.request.method !== "GET") return;

  // Network-first strategy so fresh code is always loaded when online
  event.respondWith(
    fetch(event.request)
      .then(response => {
        if (response && response.status === 200) {
          const respClone = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(event.request, respClone));
        }
        return response;
      })
      .catch(() => {
        return caches.match(event.request).then(cached => cached || caches.match("/"));
      })
  );
});
