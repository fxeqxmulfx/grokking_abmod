// Touch devices keep smaller software surfaces; hidden views release them.
export const compactRendering = typeof navigator === 'object' && (navigator.maxTouchPoints > 0 || /Android|iPhone|iPad/.test(navigator.userAgent));
export function displayScale(width, height) {
    return Math.min(devicePixelRatio || 1, compactRendering ? 1 : 2, Math.sqrt((compactRendering ? 262144 : 1048576) / (width * height)));
}
export function networkScale(width, height) {
    return Math.min(displayScale(width, height), Math.sqrt((compactRendering ? 131072 : 524288) / (width * height)));
}
//# sourceMappingURL=render-policy.js.map