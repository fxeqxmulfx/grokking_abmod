const P = 97;
const colorStops = [0, .35, .7, 1];
const positiveColors = [[220, 190, 190], [224, 112, 151], [181, 38, 111], [83, 13, 65]];
const negativeColors = [[187, 207, 217], [96, 178, 205], [36, 110, 176], [18, 38, 97]];
export function weightGradient(positive) {
    const colors = positive ? positiveColors : negativeColors;
    return `linear-gradient(90deg, ${colors.map((rgb, i) => `rgb(${rgb.join(',')}) ${100 * colorStops[i]}%`).join(', ')})`;
}
export function weightMagnitude(value, range) {
    const strength = Math.log1p(Math.abs(value) / Math.max(range, 1e-9));
    return strength / (1 + strength);
}
function weightRGB(positive, magnitude) {
    const colors = positive ? positiveColors : negativeColors;
    for (let i = 1; i < colorStops.length; i++) {
        if (magnitude > colorStops[i])
            continue;
        const fraction = (magnitude - colorStops[i - 1]) / (colorStops[i] - colorStops[i - 1]);
        const component = (c) => Math.round(colors[i - 1][c] + fraction * (colors[i][c] - colors[i - 1][c]));
        return [component(0), component(1), component(2)];
    }
    return colors.at(-1);
}
export function weightColor(positive, magnitude) {
    return `rgb(${weightRGB(positive, magnitude).join(',')})`;
}
export function weightPalette(levels) {
    const palette = new Uint8ClampedArray(2 * levels * 4);
    for (let sign = 0; sign < 2; sign++)
        for (let i = 0; i < levels; i++) {
            const at = (sign * levels + i) * 4;
            palette.set(weightRGB(sign === 0, i / (levels - 1)), at);
            palette[at + 3] = 255;
        }
    return palette;
}
const initialRanges = new WeakMap();
export function initialWeightRanges(initial, channels) {
    const cached = initialRanges.get(initial);
    if (cached?.channels === channels)
        return cached.ranges;
    const e = P * channels;
    let embeddingSquares = 0, matrixSquares = 0;
    for (let i = 0; i < initial.length; i++) {
        const square = initial[i] * initial[i];
        if (i < e)
            embeddingSquares += square;
        else
            matrixSquares += square;
    }
    // Both views use initial RMS, so later epochs and outliers cannot rescale colors.
    const ranges = [Math.max(1e-9, Math.sqrt(embeddingSquares / e)), Math.max(1e-9, Math.sqrt(matrixSquares / (channels * channels)))];
    initialRanges.set(initial, { channels, ranges });
    return ranges;
}
//# sourceMappingURL=weight-colors.js.map