/**
 * Copy text and toast the outcome.
 *
 * `navigator.clipboard` is absent over plain HTTP and in some embedded views,
 * so the optional chain matters — one guarded path instead of every caller
 * remembering it (and one caller forgetting used to throw).
 */
export function copyToClipboard(
  text: string,
  toast: (message: string, tone?: "info" | "warn" | "error") => void,
  label = "address"
): void {
  const clip = typeof navigator !== "undefined" ? navigator.clipboard : undefined;
  if (!clip) {
    toast("clipboard unavailable", "warn");
    return;
  }
  clip.writeText(text).then(
    () => toast(`${label} copied`),
    () => toast(`could not copy ${label}`, "error")
  );
}
