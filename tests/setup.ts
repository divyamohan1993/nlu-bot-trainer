/**
 * Vitest global test setup
 *
 * Provides:
 * - localStorage mock for Node.js environment
 * - performance.now() polyfill
 * - Common test utilities
 */

// Mock localStorage for tests that use the store
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] ?? null,
    setItem: (key: string, value: string) => { store[key] = value; },
    removeItem: (key: string) => { delete store[key]; },
    clear: () => { store = {}; },
    get length() { return Object.keys(store).length; },
    key: (index: number) => Object.keys(store)[index] ?? null,
  };
})();

if (typeof globalThis.localStorage === "undefined") {
  Object.defineProperty(globalThis, "localStorage", { value: localStorageMock });
}

// Ensure performance.now is available
if (typeof globalThis.performance === "undefined") {
  Object.defineProperty(globalThis, "performance", {
    value: { now: () => Date.now() },
  });
}
