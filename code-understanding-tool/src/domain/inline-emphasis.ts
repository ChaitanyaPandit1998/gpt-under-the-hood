type Style = "text" | "bold" | "italic" | "bold-italic";
type Part = { style: Style; text: string; start: number; end: number };

// Only paired asterisks are formatting. Leave code, escapes, and operators literal.
const spans = /(?<!`)(?<ticks>`+)(?!`)[\s\S]*?(?<!`)\k<ticks>(?!`)|\\\*|(?<![\\*\p{L}\p{N}])(?<mark>\*{1,3})(?<phrase>[^\s*](?:[^*\r\n]*?[^\s*])?)(?<!\\)\k<mark>(?![*\p{L}\p{N}])/gu;

export function parseInlineEmphasis(text: string): Part[] {
  const parts: Part[] = [];
  let start = 0;
  for (const match of text.matchAll(spans)) {
    const { mark, phrase } = match.groups!;
    if (!mark || !phrase) continue;
    const end = match.index + match[0].length;
    if (match.index > start) parts.push({ style: "text", text: text.slice(start, match.index), start, end: match.index });
    const style = mark.length === 3 ? "bold-italic" : mark.length === 2 ? "bold" : "italic";
    parts.push({ style, text: phrase, start: match.index, end });
    start = end;
  }
  if (start < text.length) parts.push({ style: "text", text: text.slice(start), start, end: text.length });
  return parts;
}

export function stripInlineEmphasis(text: string): string {
  return parseInlineEmphasis(text).map((part) => part.text).join("");
}
