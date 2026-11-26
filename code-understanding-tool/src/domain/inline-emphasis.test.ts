import { expect, it } from "vitest";
import { parseInlineEmphasis, stripInlineEmphasis } from "./inline-emphasis";

it("recognizes bold, italic, and combined emphasis without changing the words", () => {
  const text = "**Apply** the *limit*, even when ***busy***.";
  expect(parseInlineEmphasis(text).filter(part => part.style !== "text").map(({ style, text }) => [style, text]))
    .toEqual([["bold", "Apply"], ["italic", "limit"], ["bold-italic", "busy"]]);
  expect(stripInlineEmphasis(text)).toBe("Apply the limit, even when busy.");
});

it("leaves multiplication, inline code, escaped or unmatched markers literal", () => {
  const text = "x*y*z and 2 * 3; `**code**`; ``a ` **b**``; \\*escaped* and **unclosed.";
  expect(stripInlineEmphasis(text)).toBe(text);
  const fenced = "```txt\n**literal**\n```";
  expect(stripInlineEmphasis(fenced)).toBe(fenced);
});
