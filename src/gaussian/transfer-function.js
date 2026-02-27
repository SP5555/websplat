"use strict";

function hexToRgba(hex) {
    hex = hex.replace(/^#/, "");

    if (hex.length === 3) {
        // #RGB -> #RRGGBB, alpha=1
        hex = hex.split("").map(x => x + x).join("") + "ff";
    } else if (hex.length === 6) {
        // #RRGGBB -> #RRGGBBFF
        hex += "ff";
    } else if (hex.length !== 8) {
        throw new Error(`Invalid hex color: ${hex}`);
    }

    const r = parseInt(hex.slice(0, 2), 16) / 255;
    const g = parseInt(hex.slice(2, 4), 16) / 255;
    const b = parseInt(hex.slice(4, 6), 16) / 255;
    const a = parseInt(hex.slice(6, 8), 16) / 255;

    return [r, g, b, a];
}

function lerp(a, b, t) {
    return a + (b - a) * t;
}

export function createTransferFunction(stops) {
    stops.sort((a, b) => a.t - b.t);

    return function evaluate(t) {
        t = Math.max(0, Math.min(1, t));

        // below first stop
        if (t <= stops[0].t) return hexToRgba(stops[0].color);

        // above last stop
        if (t >= stops[stops.length - 1].t) return hexToRgba(stops[stops.length - 1].color);

        // between stops
        for (let i = 0; i < stops.length - 1; i++) {
            const a = stops[i];
            const b = stops[i + 1];

            if (t >= a.t && t <= b.t) {
                const localT = (t - a.t) / (b.t - a.t);

                const [r0, g0, b0, a0] = hexToRgba(a.color);
                const [r1, g1, b1, a1] = hexToRgba(b.color);

                return [
                    lerp(r0, r1, localT),
                    lerp(g0, g1, localT),
                    lerp(b0, b1, localT),
                    lerp(a0, a1, localT)
                ];
            }
        }
    };
}
