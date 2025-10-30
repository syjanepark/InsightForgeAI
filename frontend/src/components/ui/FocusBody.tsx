export function focusElement(selector: string) {
  const el = document.querySelector(selector);
  if (!el) return;

  // Smooth scroll into view
  el.scrollIntoView({ behavior: "smooth", block: "center" });

  // Add focus glow
  el.classList.add("focus-glow");

  // Remove after animation
  setTimeout(() => {
    el.classList.remove("focus-glow");
  }, 1500);
}
