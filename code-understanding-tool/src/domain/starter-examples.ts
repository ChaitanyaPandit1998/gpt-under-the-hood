import type { Mode } from "./explanation-types";

export type StarterExample = {
  mode: Mode;
  category: string;
  title: string;
  description: string;
  icon: string;
  input: string;
};

export const starterExamples: readonly StarterExample[] = [
  {
    mode: "code",
    category: "Code snippet",
    title: "Retry a failed request",
    description: "Follow a short TypeScript function through success and failure.",
    icon: "</>",
    input: `async function retry<T>(operation: () => Promise<T>, attempts = 3): Promise<T> {
  try {
    return await operation();
  } catch (error) {
    if (attempts <= 1) throw error;
    return retry(operation, attempts - 1);
  }
}`,
  },
  {
    mode: "pull_request",
    category: "Long pull request",
    title: "Explore a FastAPI migration",
    description: "Browse a large public change as a chapter-by-chapter study.",
    icon: "PR",
    input: "https://github.com/fastapi/fastapi/pull/9816",
  },
  {
    mode: "docs",
    category: "Technical concept",
    title: "Understand cache-aside",
    description: "Trace what happens on a cache hit, a miss, and after an update.",
    icon: "Aa",
    input: `# Cache-aside

When the application needs a record, it first checks the cache. A hit returns the cached record without reading the database. On a miss, the application reads the database, puts the result in the cache, and returns it.

After updating the database, the application can invalidate the cached record so that a later read can fetch the new value. A time-to-live (TTL) limits how long an entry stays cached. Cache-aside on its own does not guarantee that concurrent readers always see the latest value.`,
  },
];
