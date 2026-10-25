const superscripts: Record<string, string> = {
  "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
  "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
};

/** Converts common inline TeX emitted in plain-text answers into readable Unicode math. */
export function formatMathText(value: string): string {
  return value
    .replace(/\\\(|\\\)|\\\[|\\\]/g, "")
    .replace(/\\sqrt\{([^{}]*)\}/g, "√($1)")
    .replace(/\\frac\{([^{}]+)\}\{([^{}]+)\}/g, "($1)/($2)")
    .replace(/\\(?:cdot|times)\b/g, "×")
    .replace(/\\neq\b/g, "≠")
    .replace(/\\(?:leq|le)\b/g, "≤")
    .replace(/\\(?:geq|ge)\b/g, "≥")
    .replace(/\\pm\b/g, "±")
    .replace(/\\to\b/g, "→")
    .replace(/\\infty\b/g, "∞")
    .replace(/\\pi\b/g, "π")
    .replace(/([A-Za-z0-9)])\^\{([^{}]+)\}/g, (_match, base: string, exponent: string) => {
      const formatted = [...exponent].map((character) => superscripts[character] ?? character).join("");
      return `${base}${formatted}`;
    })
    .replace(/([A-Za-z0-9)])\^([0-9])/g, (_match, base: string, exponent: string) => `${base}${superscripts[exponent]}`);
}
