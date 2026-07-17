document.addEventListener("DOMContentLoaded", () => {
  const panels = { ocr: document.getElementById("ocr-panel"), detect: document.getElementById("detect-panel") };
  document.querySelectorAll(".tab").forEach(tab => tab.addEventListener("click", () => {
    const mode = tab.dataset.mode;
    Object.entries(panels).forEach(([name, panel]) => panel.classList.toggle("hidden", name !== mode));
    document.querySelectorAll(".tab").forEach(button => button.classList.toggle("active", button === tab));
  }));
  document.querySelectorAll("[data-upload-form]").forEach(form => {
    const input = form.querySelector("[data-image-input]"), preview = form.querySelector("[data-preview]"), filename = form.querySelector("[data-filename]");
    input.addEventListener("change", () => { const file = input.files[0]; filename.textContent = file ? file.name : "No file selected"; if (file) { preview.src = URL.createObjectURL(file); preview.classList.remove("hidden"); } });
    form.addEventListener("reset", () => setTimeout(() => { filename.textContent = "No file selected"; preview.src = ""; preview.classList.add("hidden"); }, 0));
    form.addEventListener("submit", () => { const button = form.querySelector(".primary"); button.textContent = "Processing..."; button.disabled = true; });
  });
  const copy = document.getElementById("copy-text");
  if (copy) copy.addEventListener("click", async () => { await navigator.clipboard.writeText(document.getElementById("extracted-text").value); copy.textContent = "Copied!"; setTimeout(() => copy.textContent = "Copy Text", 1500); });
});
