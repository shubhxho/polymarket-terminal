"use client";

/**
 * Shared motion language. The terminal's visual identity is calm and dense —
 * ami.dev restraint, not a marketing site — so animation here is a courier of
 * *meaning*, never decoration: it signals that a screen changed, that a panel
 * arrived, that a row entered the tape. Durations are short and easing is
 * near-linear so a fast market never strobes.
 *
 * Everything is exported from one place so every surface animates the same
 * way. Reduced-motion is honoured globally by wrapping the tree in
 * `<MotionConfig reducedMotion="user">` (see `Providers`), which collapses
 * transform/opacity animations for users who ask for it — individual call
 * sites do not restate the media query.
 */
import type { Transition, Variants } from "motion/react";

/** House easing: a soft ease-out that settles without overshoot. */
export const EASE = [0.22, 0.61, 0.36, 1] as const;

/** The default transition for chrome — fast enough to feel instant. */
export const transition: Transition = { duration: 0.18, ease: EASE };

/** Slightly longer, for larger surfaces (a whole screen crossfading). */
export const transitionSlow: Transition = { duration: 0.24, ease: EASE };

/**
 * Screen swap. Screens live in an `AnimatePresence` keyed on the screen
 * identity; the outgoing one fades and drops a few px while the incoming one
 * rises into place. `mode="popLayout"` in the host keeps them from reflowing
 * each other mid-flight.
 */
export const screenVariants: Variants = {
  initial: { opacity: 0, y: 6 },
  animate: { opacity: 1, y: 0, transition: transitionSlow },
  exit: { opacity: 0, y: -4, transition: transition },
};

/** A panel or card arriving — a gentle fade-and-lift. */
export const panelVariants: Variants = {
  initial: { opacity: 0, y: 8 },
  animate: { opacity: 1, y: 0, transition },
  exit: { opacity: 0, transition },
};

/**
 * Stagger container: children animate in sequence. Kept tight (24ms) so a
 * 40-row grid finishes revealing well under a quarter second.
 */
export const staggerContainer: Variants = {
  initial: {},
  animate: { transition: { staggerChildren: 0.024, delayChildren: 0.02 } },
  exit: {},
};

/** A single row/list-item entering — pairs with `staggerContainer`. */
export const rowVariants: Variants = {
  initial: { opacity: 0, y: 4 },
  animate: { opacity: 1, y: 0, transition },
  exit: { opacity: 0, transition: { duration: 0.12, ease: EASE } },
};

/** A print entering the tape from the top: brief slide + fade. */
export const tapeRowVariants: Variants = {
  initial: { opacity: 0, y: -6 },
  animate: { opacity: 1, y: 0, transition },
  exit: { opacity: 0, transition: { duration: 0.12, ease: EASE } },
};

/** Toast / popover: rise and fade, snappy on the way out. */
export const popVariants: Variants = {
  initial: { opacity: 0, y: 10, scale: 0.98 },
  animate: { opacity: 1, y: 0, scale: 1, transition },
  exit: { opacity: 0, y: 6, scale: 0.98, transition: { duration: 0.14, ease: EASE } },
};

/** Overlay scrim (command palette backdrop). */
export const scrimVariants: Variants = {
  initial: { opacity: 0 },
  animate: { opacity: 1, transition: { duration: 0.14, ease: EASE } },
  exit: { opacity: 0, transition: { duration: 0.12, ease: EASE } },
};

/** Tap feedback for interactive chrome — a hair of scale, no bounce. */
export const tapScale = { scale: 0.97 } as const;
