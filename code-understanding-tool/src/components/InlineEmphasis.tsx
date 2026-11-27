import { parseInlineEmphasis } from "../domain/inline-emphasis";

export function InlineEmphasis({ text }: { text: string }) {
  return parseInlineEmphasis(text).map((part) => {
    switch (part.style) {
      case "bold": return <strong key={part.start}>{part.text}</strong>;
      case "italic": return <em key={part.start}>{part.text}</em>;
      case "bold-italic": return <strong key={part.start}><em>{part.text}</em></strong>;
      default: return part.text;
    }
  });
}
