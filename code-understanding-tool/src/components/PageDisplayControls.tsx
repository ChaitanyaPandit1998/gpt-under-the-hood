import { themeLabels, themeModes, type ThemeMode } from "../domain/theme";

type Props = {
  zoom: number;
  theme: ThemeMode;
  onZoomChange: (update: (zoom: number) => number) => void;
  onThemeChange: (theme: ThemeMode) => void;
};

export function PageDisplayControls({ zoom, theme, onZoomChange, onThemeChange }: Props) {
  return <div className="page-display-controls" aria-label="Page display settings">
    <div className="page-zoom-controls" role="group" aria-label="Page zoom">
      <button type="button" onClick={() => onZoomChange((value) => Math.max(80, value - 10))} disabled={zoom <= 80} aria-label="Zoom out" title="Zoom out">−</button>
      <output aria-live="polite" aria-label="Current page zoom">{zoom}%</output>
      <button type="button" onClick={() => onZoomChange((value) => Math.min(130, value + 10))} disabled={zoom >= 130} aria-label="Zoom in" title="Zoom in">+</button>
      <button type="button" onClick={() => onZoomChange(() => 100)} disabled={zoom === 100} aria-label="Reset zoom" title="Reset zoom">↺</button>
    </div>
    <label className="page-theme-select">
      <span className={`theme-swatch theme-swatch-${theme}`} aria-hidden="true" />
      <span className="visually-hidden">Color theme</span>
      <select value={theme} onChange={(event) => onThemeChange(event.target.value as ThemeMode)} aria-label="Color theme">
        {themeModes.map((mode) => <option key={mode} value={mode}>{themeLabels[mode]} theme</option>)}
      </select>
    </label>
  </div>;
}
