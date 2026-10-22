type Props = {
  zoom: number;
  theme: "light" | "dark";
  onZoomChange: (update: (zoom: number) => number) => void;
  onThemeChange: (update: (theme: "light" | "dark") => "light" | "dark") => void;
};

export function PageDisplayControls({ zoom, theme, onZoomChange, onThemeChange }: Props) {
  return <div className="page-display-controls" aria-label="Page display settings">
    <div className="page-zoom-controls" role="group" aria-label="Page zoom">
      <button type="button" onClick={() => onZoomChange((value) => Math.max(80, value - 10))} disabled={zoom <= 80} aria-label="Zoom out" title="Zoom out">−</button>
      <output aria-live="polite" aria-label="Current page zoom">{zoom}%</output>
      <button type="button" onClick={() => onZoomChange((value) => Math.min(130, value + 10))} disabled={zoom >= 130} aria-label="Zoom in" title="Zoom in">+</button>
      <button type="button" onClick={() => onZoomChange(() => 100)} disabled={zoom === 100} aria-label="Reset zoom" title="Reset zoom">↺</button>
    </div>
    <button className="page-theme-toggle" type="button" onClick={() => onThemeChange((value) => value === "light" ? "dark" : "light")} aria-pressed={theme === "dark"}>
      <span aria-hidden="true">{theme === "dark" ? "☀" : "◐"}</span>{theme === "dark" ? "Light theme" : "Dark theme"}
    </button>
  </div>;
}
