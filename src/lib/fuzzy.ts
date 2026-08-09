/**
 * Subsequence fuzzy matching for the command palette.
 *
 * Deliberately not a generic edit-distance: what matters in a palette is that
 * typing the initials of a long name ranks it first ("npm" → "Next Prime
 * Minister"), and that a match at a word boundary beats one buried mid-token.
 * Levenshtein gets both of those wrong.
 */

export type FuzzyMatch = {
  score: number;
  /** Indices of `text` that matched, for highlighting. */
  positions: number[];
};

const SCORE_START = 12; // match at the very beginning of the string
const SCORE_WORD = 9; // match at the start of a word
const SCORE_CONSECUTIVE = 6; // immediately after the previous match
const SCORE_MATCH = 1; // any match at all
const PENALTY_GAP = -0.4; // per skipped character

function isBoundary(ch: string): boolean {
  return ch === " " || ch === "-" || ch === "_" || ch === "/" || ch === ":" || ch === ".";
}

/**
 * Scores `query` against `text`. Returns `null` when `query` is not a
 * subsequence of `text`, so callers can filter and rank in one pass.
 */
export function fuzzyMatch(query: string, text: string): FuzzyMatch | null {
  if (!query) return { score: 0, positions: [] };

  const q = query.toLowerCase();
  const t = text.toLowerCase();
  const positions: number[] = [];

  let score = 0;
  let ti = 0;
  let lastMatch = -2;

  for (let qi = 0; qi < q.length; qi++) {
    const ch = q[qi];
    // Spaces in the query are separators, not characters to find.
    if (ch === " ") continue;

    let found = -1;
    for (let i = ti; i < t.length; i++) {
      if (t[i] !== ch) continue;
      found = i;
      break;
    }
    if (found === -1) return null;

    if (found === 0) score += SCORE_START;
    else if (isBoundary(t[found - 1])) score += SCORE_WORD;
    else if (found === lastMatch + 1) score += SCORE_CONSECUTIVE;
    else score += SCORE_MATCH;

    if (found > ti) score += PENALTY_GAP * (found - ti);

    positions.push(found);
    lastMatch = found;
    ti = found + 1;
  }

  // Shorter targets are better matches for the same query — "MON" should beat
  // "Market Monitor Overview" when the user typed "mon".
  score += Math.max(0, 20 - text.length) * 0.15;
  return { score, positions };
}

/** Splits `text` into alternating unmatched / matched runs for rendering. */
export function highlight(text: string, positions: readonly number[]) {
  const set = new Set(positions);
  const parts: { text: string; hit: boolean }[] = [];
  let buf = "";
  let mode = false;

  for (let i = 0; i < text.length; i++) {
    const hit = set.has(i);
    if (hit !== mode && buf) {
      parts.push({ text: buf, hit: mode });
      buf = "";
    }
    mode = hit;
    buf += text[i];
  }
  if (buf) parts.push({ text: buf, hit: mode });
  return parts;
}
