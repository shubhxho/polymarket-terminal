import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Conditional class composition, the modern standard.
 *
 * `clsx` flattens conditionals and arrays into a class string; `twMerge` then
 * resolves Tailwind conflicts so a later `text-accent` cleanly overrides an
 * earlier `text-muted` instead of both landing in the DOM and leaving the
 * winner to specificity. Together they replace the hand-rolled template-literal
 * ternaries the terminal used to concatenate classes with.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
