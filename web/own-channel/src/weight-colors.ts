import type {Weights} from './types.ts';
const P = 97;
type RGB = readonly [number, number, number];
const colorStops = [0, .35, .7, 1] as const;
const positiveColors: readonly RGB[] = [[220, 190, 190], [224, 112, 151], [181, 38, 111], [83, 13, 65]];
const negativeColors: readonly RGB[] = [[187, 207, 217], [96, 178, 205], [36, 110, 176], [18, 38, 97]];

export function weightGradient(positive: boolean): string {
  const colors = positive ? positiveColors : negativeColors;
  return `linear-gradient(90deg, ${colors.map((rgb, i) => `rgb(${rgb.join(',')}) ${100 * colorStops[i]}%`).join(', ')})`;
}
export function weightMagnitude(value: number, range: number): number {
  const strength = Math.log1p(Math.abs(value) / Math.max(range, 1e-9));
  return strength / (1 + strength);
}
function weightRGB(positive: boolean, magnitude: number): RGB {
  const colors = positive ? positiveColors : negativeColors;
  for (let i = 1; i < colorStops.length; i++) {
    if (magnitude > colorStops[i]) continue;
    const fraction = (magnitude - colorStops[i - 1]) / (colorStops[i] - colorStops[i - 1]);
    const component = (c: number) => Math.round(colors[i - 1][c] + fraction * (colors[i][c] - colors[i - 1][c]));
    return [component(0), component(1), component(2)];
  }
  return colors.at(-1)!;
}
export function weightColor(positive: boolean, magnitude: number): string {
  return `rgb(${weightRGB(positive, magnitude).join(',')})`;
}
export function weightPalette(levels: number): Uint8ClampedArray {
  const palette = new Uint8ClampedArray(2 * levels * 4);
  for (let sign = 0; sign < 2; sign++) for (let i = 0; i < levels; i++) {
    const at = (sign * levels + i) * 4;
    palette.set(weightRGB(sign === 0, i / (levels - 1)), at); palette[at + 3] = 255;
  }
  return palette;
}
const initialRanges = new WeakMap<Weights, {channels: number; ranges: [number, number]}>();
export function initialWeightRanges(initial: Weights, channels: number): [number, number] {
  const cached = initialRanges.get(initial);
  if (cached?.channels === channels) return cached.ranges;
  const e = P * channels; let embeddingSquares = 0, matrixSquares = 0;
  for (let i = 0; i < initial.length; i++) {
    const square = initial[i] * initial[i];
    if (i < e) embeddingSquares += square; else matrixSquares += square;
  }
  // Both views use initial RMS, so later epochs and outliers cannot rescale colors.
  const ranges: [number, number] = [Math.max(1e-9, Math.sqrt(embeddingSquares / e)), Math.max(1e-9, Math.sqrt(matrixSquares / (channels * channels)))];
  initialRanges.set(initial, {channels, ranges});
  return ranges;
}
