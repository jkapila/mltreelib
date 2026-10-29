// Click-to-zoom for documentation images (self-contained lightbox).
document.addEventListener("DOMContentLoaded", function () {
  var overlay = document.createElement("div");
  overlay.style.cssText =
    "position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;" +
    "z-index:9999;cursor:zoom-out;overflow:auto";
  var big = document.createElement("img");
  big.style.cssText =
    "max-width:96vw;max-height:96vh;display:block;margin:2vh auto;" +
    "border-radius:6px;background:#fff";
  overlay.appendChild(big);
  document.body.appendChild(overlay);
  overlay.addEventListener("click", function () {
    overlay.style.display = "none";
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") overlay.style.display = "none";
  });
  var imgs = document.querySelectorAll(".md-content img, article img");
  imgs.forEach(function (el) {
    el.style.cursor = "zoom-in";
    el.addEventListener("click", function () {
      big.src = el.currentSrc || el.src;
      overlay.style.display = "block";
    });
  });
});
