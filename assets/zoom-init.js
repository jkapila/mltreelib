// Standard docs zoom (medium-zoom, vendored) - every image on the site is
// click-to-zoom: plots, tree diagrams, tables of figures.
document.addEventListener("DOMContentLoaded", function () {
  mediumZoom(document.querySelectorAll(".md-content img, article img, main img"), {
    background: "rgba(0,0,0,.85)",
  });
});
