"use client";

import { useEffect, useState } from "react";

function utcNow(): string {
  return `${new Date().toISOString().slice(0, 19).replace("T", " ")} UTC`;
}

export function Clock() {
  const [now, setNow] = useState<string | null>(null);

  useEffect(() => {
    setNow(utcNow());
    const id = setInterval(() => setNow(utcNow()), 1000);
    return () => clearInterval(id);
  }, []);

  // Render a fixed-width placeholder until mounted to avoid hydration mismatch
  return <span className="tabular-nums text-foreground">{now ?? "···· ·· ·· ··:··:·· UTC"}</span>;
}
