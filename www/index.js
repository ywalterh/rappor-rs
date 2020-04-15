import * as wasm from "fuzzy-men";

const submit_btn = document.getElementById("submit_btn");

submit_btn.addEventListener("click", () => {
  wasm.greet(document.getElementById("test1").value);
});
