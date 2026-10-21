// The theme's download button links the notebook source without the HTML `download` attribute,
// so the browser shows the JSON. Same origin, so the attribute turns the click into a file save.
document.addEventListener("DOMContentLoaded", () => {
  for (const a of document.querySelectorAll("a.btn-download-source-button")) {
    a.setAttribute("download", a.pathname.split("/").pop());
    a.removeAttribute("target");
  }
});
