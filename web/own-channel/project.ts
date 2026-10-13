// Build tools work from either their TypeScript sources or compiled build/ files.
const here = new URL('./', import.meta.url);
export const root = here.pathname.endsWith('/build/') ? new URL('../', here) : here;
