"use client";

import { useEffect } from "react";

export function SearchFocuser() {
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement).tagName;
      if (e.key === "/" && tag !== "INPUT" && tag !== "TEXTAREA") {
        e.preventDefault();
        document.querySelector<HTMLInputElement>("[data-search]")?.focus();
      }
      if (e.key === "Escape") {
        const el = document.querySelector<HTMLInputElement>("[data-search]");
        if (document.activeElement === el) el?.blur();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  return null;
}
