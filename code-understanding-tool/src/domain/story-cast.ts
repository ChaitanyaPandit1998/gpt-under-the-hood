export const storyHeadChoices = [
  { id: "short-hair", label: "Short dark hair", skin: "#C88565" },
  { id: "long-hair", label: "Long dark hair", skin: "#F1AB8B" },
] as const;

export type StoryHeadId = typeof storyHeadChoices[number]["id"];
export type StoryCast = Partial<Record<StoryHeadId, string>>;
export const storyCastNameLimit = 48;

export function parseStoryCast(value: unknown): StoryCast | undefined {
  if (value === undefined) return undefined;
  const invalid = () => { throw new Error("Enter a distinct fictional name or role for each face."); };
  if (!value || typeof value !== "object" || Array.isArray(value)) return invalid();
  const entries = Object.entries(value);
  const seen = new Set<string>();
  if (!entries.length) return invalid();
  for (const [id, name] of entries) {
    if (!storyHeadChoices.some(choice => choice.id === id) || typeof name !== "string" ||
      name !== name.trim() || !name || name.length > storyCastNameLimit || /[\p{Cc}\p{Cf}<>]/u.test(name)) return invalid();
    const folded = name.normalize("NFKC").toLowerCase();
    if (seen.has(folded)) return invalid();
    seen.add(folded);
  }
  return Object.fromEntries(entries);
}

export function lockedStorySlots(cast: StoryCast) {
  return storyHeadChoices.flatMap(({ id: asset, label, skin }, index) => {
    const role = cast[asset];
    return role ? [{ slot: `cast-${index + 1}`, asset, label, skin, role }] : [];
  });
}
