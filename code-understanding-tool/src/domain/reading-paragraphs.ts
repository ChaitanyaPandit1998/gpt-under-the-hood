import { parseInlineEmphasis } from "./inline-emphasis";

const sentences = typeof Intl.Segmenter === "function" ? new Intl.Segmenter(undefined, { granularity: "sentence" }) : undefined;

export function splitReadingParagraphs(text: string): string[] {
  const normalized = text.trim().replace(/\r\n?/g, "\n");
  if (!normalized) return [];
  if (/^(?: {0,3})(?:```|~~~)/m.test(normalized)) return [normalized];

  return normalized.split(/\n[ \t]*\n(?:[ \t]*\n)*/).flatMap((block) => {
    const prose = block.trim();
    if (!prose) return [];
    // Respect authored line breaks. Split only long, uninterrupted prose.
    if (!sentences || prose.length < 520 || prose.includes("\n")) return [prose];

    const paragraphs: string[] = [];
    const emphasized = parseInlineEmphasis(prose).filter(part => part.style !== "text");
    let start = 0;
    for (const { index } of sentences.segment(prose)) {
      if (index - start < 350 || prose.length - index < 150) continue;
      if (prose.slice(start, index).split("`").length % 2 === 0) continue;
      if (emphasized.some(part => part.start < index && index < part.end)) continue;
      paragraphs.push(prose.slice(start, index).trim());
      start = index;
    }
    paragraphs.push(prose.slice(start).trim());
    return paragraphs;
  });
}
