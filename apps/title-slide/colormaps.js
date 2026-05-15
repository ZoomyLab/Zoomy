/**
 * apps/title-slide/colormaps.js — 256-entry LUTs for the WebGL render shader.
 *
 * Each LUT is a 256 × 4 Uint8Array (RGBA). Index 0 is the "wall /
 * background / dry" colour; indices 1..255 sweep low → high field value.
 * The render shader samples ``texture(uLUT, vec2((v+1)/256, 0.5))`` with
 * ``v ∈ [0, 254]``, so index 1 is the lowest fluid value and index 255
 * is saturation.
 */

function hex(s) {
  return [
    parseInt(s.slice(1, 3), 16),
    parseInt(s.slice(3, 5), 16),
    parseInt(s.slice(5, 7), 16),
  ];
}

function lerp(a, b, t) {
  return [
    Math.round(a[0] + (b[0] - a[0]) * t),
    Math.round(a[1] + (b[1] - a[1]) * t),
    Math.round(a[2] + (b[2] - a[2]) * t),
  ];
}

/** Build a 256-entry LUT from a sorted list of [position, "#rrggbb"] stops. */
function gradient(stops, opts = {}) {
  const alpha = opts.alpha ?? 255;
  const bg = opts.bg ?? stops[0][1];
  const out = new Uint8Array(256 * 4);
  const bgRGB = hex(bg);
  out[0] = bgRGB[0]; out[1] = bgRGB[1]; out[2] = bgRGB[2]; out[3] = alpha;
  for (let i = 1; i < 256; i++) {
    const t = (i - 1) / 254;
    let a = stops[0], b = stops[stops.length - 1];
    for (let k = 0; k < stops.length - 1; k++) {
      if (t >= stops[k][0] && t <= stops[k + 1][0]) { a = stops[k]; b = stops[k + 1]; break; }
    }
    const u = b[0] > a[0] ? (t - a[0]) / (b[0] - a[0]) : 0;
    const rgb = lerp(hex(a[1]), hex(b[1]), u);
    out[i * 4] = rgb[0]; out[i * 4 + 1] = rgb[1]; out[i * 4 + 2] = rgb[2]; out[i * 4 + 3] = alpha;
  }
  return out;
}

export const COLORMAPS = {
  // Default: deep navy through ocean blues to a soft cyan highlight —
  // reads as "water" without being overly saturated on a projector.
  water: () => gradient([
    [0.00, "#08233a"],
    [0.30, "#11507a"],
    [0.60, "#2b8fc7"],
    [0.85, "#7ec8e3"],
    [1.00, "#e6f6fb"],
  ]),
  // Cool, very dark base — good when overlaying light text.
  deepwater: () => gradient([
    [0.00, "#020912"],
    [0.40, "#093058"],
    [0.75, "#1f6aa0"],
    [1.00, "#a8d8ee"],
  ]),
  // Inverse — pale base, dark crests; useful for light slide themes.
  ice: () => gradient([
    [0.00, "#f3f8fc"],
    [0.40, "#bfdbed"],
    [0.75, "#5e8fb6"],
    [1.00, "#1a3b5e"],
  ]),
  viridis: () => gradient([
    [0.00, "#440154"],
    [0.25, "#3b528b"],
    [0.50, "#21918c"],
    [0.75, "#5ec962"],
    [1.00, "#fde725"],
  ]),
  magma: () => gradient([
    [0.00, "#000004"],
    [0.25, "#3b0f70"],
    [0.50, "#8c2981"],
    [0.75, "#de4968"],
    [1.00, "#fcfdbf"],
  ]),
  grayscale: () => gradient([
    [0.00, "#000000"],
    [1.00, "#ffffff"],
  ]),
  // Warm sunset — works well over dark slide backgrounds.
  sunset: () => gradient([
    [0.00, "#0d1b2a"],
    [0.40, "#7a3c2a"],
    [0.75, "#f08a3e"],
    [1.00, "#fde7c4"],
  ]),
};

export function buildLUT(name) {
  const fn = COLORMAPS[name] || COLORMAPS.water;
  return fn();
}

export const COLORMAP_NAMES = Object.keys(COLORMAPS);
