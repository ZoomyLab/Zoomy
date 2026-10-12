// Tag filter of docs/book/tutorials/index.md (generate_tutorial_index.py).
// A card's tags: data-tags of its .zt-data span, copied onto the grid column.
// Selected chips: OR within a category, AND across categories.
document.addEventListener("DOMContentLoaded", function () {
  var filter = document.querySelector(".zt-filter");
  if (!filter) return;
  var cols = [];
  document.querySelectorAll(".zt-data").forEach(function (span) {
    var col = span.closest(".sd-col") || span.closest(".sd-card");
    col.dataset.tags = span.dataset.tags;
    cols.push(col);
  });
  var chips = filter.querySelectorAll(".zt-chip");
  var empty = document.querySelector(".zt-empty");
  function apply() {
    var want = {};
    chips.forEach(function (c) {
      if (!c.classList.contains("zt-on")) return;
      var cat = c.dataset.tag.split(":")[0];
      (want[cat] = want[cat] || []).push(c.dataset.tag);
    });
    var shown = 0;
    cols.forEach(function (col) {
      var tags = col.dataset.tags.split(" ");
      var ok = Object.keys(want).every(function (cat) {
        return want[cat].some(function (t) { return tags.indexOf(t) >= 0; });
      });
      col.hidden = !ok;
      if (ok) shown++;
    });
    if (empty) empty.hidden = shown > 0;
  }
  chips.forEach(function (c) {
    c.addEventListener("click", function () { c.classList.toggle("zt-on"); apply(); });
  });
  var clear = filter.querySelector(".zt-clear");
  if (clear) clear.addEventListener("click", function () {
    chips.forEach(function (c) { c.classList.remove("zt-on"); });
    apply();
  });
});
