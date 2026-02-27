'use strict';

import { createTransferFunction } from "../gaussian/transfer-function.js";

// custom JSON loader for meshes
// there is no general JSON format for gaussian splats

const transferFunction0 = [
    { t: 0.00, color: "#6f5fff76" },
    { t: 0.50, color: "#6439ff7c" },
    { t: 0.55, color: "#4917ff87" },
    { t: 0.60, color: "#ff9f0f9c" },
    { t: 0.80, color: "#ff990099" },
    { t: 1.00, color: "#eeff00ac" },
];

const transferFunction1 = [
    { t: 0.30, color: "#0606ff99" },
    { t: 0.60, color: "#ffffff77" },
    { t: 1.00, color: "#ff000088" },
];

const tf = createTransferFunction(transferFunction0);

export default class JSONLoader {
    constructor() {
        // hmm
    }

    async load(file) {
        const text = await file.text();
        const data = JSON.parse(text);

        if (!data.num_points || !Array.isArray(data.points)) {
            throw new Error("Invalid JSON format");
        }

        const numPoints = data.num_points;
        const points = data.points;

        /* ===== Data Inspection ===== */
        // inspect min and max mass
        let minMass = Infinity;
        let maxMass = -Infinity;
        for (let i = 0; i < points.length; i++) {
            if (points[i].mass < minMass) minMass = points[i].mass;
            if (points[i].mass > maxMass) maxMass = points[i].mass;
        }
        const logMin = Math.log10(minMass);
        const logMax = Math.log10(maxMass);
        const logRange = logMax - logMin;

        console.log(`Min mass: ${minMass}\nMax mass: ${maxMass}\n`
                  + `Log Min mass: ${logMin}\nLog Max mass: ${logMax}`);

        // const bucketCount = 20;
        // const buckets = new Array(bucketCount).fill(0);
        // for (let i = 0; i < points.length; i++) {
        //     let index = Math.floor((Math.log10(points[i].mass) - logMin) / logRange * bucketCount);
        //     if (index === bucketCount) index = bucketCount - 1;
        //     buckets[index]++;
        // }
        // console.log(`Mass distribution:`);
        // for (let i = 0; i < bucketCount; i++) {
        //     console.log(`Bucket ${i}: ${buckets[i]} points`);
        // }

        /* ============================ */

        const positions = new Float32Array(numPoints * 3);
        const scales = new Float32Array(numPoints * 3);
        const rotations = new Float32Array(numPoints * 4);
        const colors = new Float32Array(numPoints * 3);
        const opacities = new Float32Array(numPoints);

        for (let i = 0; i < numPoints; i++) {
            positions[i * 3 + 0] = points[i].position.x;
            positions[i * 3 + 1] = points[i].position.y;
            positions[i * 3 + 2] = points[i].position.z;

            const radius = points[i].radius;
            scales[i * 3 + 0] = radius;
            scales[i * 3 + 1] = radius;
            scales[i * 3 + 2] = radius;

            // identity quaternion
            rotations[i * 4 + 0] = 0;
            rotations[i * 4 + 1] = 0;
            rotations[i * 4 + 2] = 0;
            rotations[i * 4 + 3] = 1;

            const logMass = Math.log10(points[i].mass);
            let t = (logMass - logMin) / logRange;

            const [r, g, b, a] = tf(t);
            colors[i * 3 + 0] = r;
            colors[i * 3 + 1] = g;
            colors[i * 3 + 2] = b;
            opacities[i] = a;
        }

        console.log(`Loaded JSON: ${numPoints} points`);

        return {
            vertexCount: numPoints,
            positions,
            scales,
            rotations,
            colors,
            opacities
        };
    }
}
