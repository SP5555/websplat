'use strict';

export class SplatDataPacker {
    static build(meshData) {
        const { splatCount, positions, covariances, colors, opacities } = meshData;

        // vec3 pos + f opacity + 6f covariance + 2f _pad + vec3 color + 1f _pad = 16 floats
        const floatsPerSplat = 16;
        const bufferData = new Float32Array(splatCount * floatsPerSplat);

        // Layout of a splat in the buffer::
        // [  px,   py,   pz, opacity,
        //   cxx,  cxy,  cxz,     ---,
        //   cyy,  cyz,  czz,     ---,
        //     r,    g,    b,     --- ]
        for (let i = 0; i < splatCount; i++) {
            const baseSrc = i * 3;
            const baseCov = i * 6;
            const baseDst = i * floatsPerSplat;

            bufferData[baseDst + 0] = positions[baseSrc + 0];
            bufferData[baseDst + 1] = positions[baseSrc + 1];
            bufferData[baseDst + 2] = positions[baseSrc + 2];
            bufferData[baseDst + 3] = 0.0;

            bufferData[baseDst + 4] = covariances[baseCov + 0];
            bufferData[baseDst + 5] = covariances[baseCov + 1];
            bufferData[baseDst + 6] = covariances[baseCov + 2];
            bufferData[baseDst + 7] = 0.0;

            bufferData[baseDst + 8]  = covariances[baseCov + 3];
            bufferData[baseDst + 9]  = covariances[baseCov + 4];
            bufferData[baseDst + 10] = covariances[baseCov + 5];
            bufferData[baseDst + 11] = 0.0;

            bufferData[baseDst + 12] = colors[baseSrc + 0];
            bufferData[baseDst + 13] = colors[baseSrc + 1];
            bufferData[baseDst + 14] = colors[baseSrc + 2];
            bufferData[baseDst + 15] = opacities[i];
        }

        return { splatCount, floatsPerSplat, bufferData };
    }
}
