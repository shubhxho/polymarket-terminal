// The `bun:test` module + test globals live in `bun-types`, which `@types/bun`
// only pulls in via a transitive `/// <reference types="bun-types" />`. The TS 7
// native compiler doesn't follow that transitive reference from an auto-included
// @types package, so we reference it directly here to type the test files.
/// <reference types="bun-types" />
