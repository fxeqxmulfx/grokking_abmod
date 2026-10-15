// Keep visible canvases sharp without retaining large hidden surfaces.
export const compactRendering = typeof navigator === 'object' && (navigator.maxTouchPoints > 0 || /Android|iPhone|iPad/.test(navigator.userAgent));
export function displayScale(width: number, height: number) {
  return Math.min(devicePixelRatio || 1, 2, Math.sqrt((compactRendering ? 524288 : 1048576) / (width * height)));
}
export function networkScale(width: number, height: number) {
  return Math.min(displayScale(width, height), Math.sqrt(524288 / (width * height)));
}
