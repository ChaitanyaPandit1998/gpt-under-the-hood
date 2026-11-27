import { splitReadingParagraphs } from "../domain/reading-paragraphs";
import { InlineEmphasis } from "./InlineEmphasis";

export function ReadingParagraphs({ text, paragraphClassName }: { text: string; paragraphClassName?: string }) {
  return <div className="reading-paragraphs">
    {splitReadingParagraphs(text).map((paragraph, index) => <p key={index} className={paragraphClassName}><InlineEmphasis text={paragraph} /></p>)}
  </div>;
}
