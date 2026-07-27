import Link from "next/link";

export default function NotFound() {
  return (
    <main className="flex flex-1 flex-col items-center justify-center gap-3">
      <p className="text-2xl font-bold text-red">404 · MARKET NOT FOUND</p>
      <p className="text-xs text-muted">
        This event does not exist or has been delisted.
      </p>
      <Link
        href="/"
        className="border border-edge bg-panel px-3 py-1.5 text-xs text-muted hover:border-edge-bright hover:text-accent"
      >
        &lt; BACK TO MARKETS
      </Link>
    </main>
  );
}
