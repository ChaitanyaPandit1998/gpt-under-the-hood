export const themeModes = ["light", "dark", "yellow", "orange", "red", "blue"] as const;

export type ThemeMode = typeof themeModes[number];

export function isThemeMode(value: string | null): value is ThemeMode {
  return themeModes.some((theme) => theme === value);
}

export const themeLabels: Record<ThemeMode, string> = {
  light: "Light",
  dark: "Dark",
  yellow: "Yellow",
  orange: "Orange",
  red: "Red",
  blue: "Blue",
};

export const themeAccents: Record<ThemeMode, { edge: string; marker: string; canvas: string }> = {
  light: { edge: "#968ce5", marker: "#8176df", canvas: "#e8e6f2" },
  dark: { edge: "#a79af7", marker: "#a79af7", canvas: "#343747" },
  yellow: { edge: "#c18a12", marker: "#a96f00", canvas: "#efe5c9" },
  orange: { edge: "#d56b2b", marker: "#bd5517", canvas: "#f0dfd3" },
  red: { edge: "#d15661", marker: "#b83a49", canvas: "#f0dfe2" },
  blue: { edge: "#4a83cf", marker: "#2f69b5", canvas: "#dce6f2" },
};
