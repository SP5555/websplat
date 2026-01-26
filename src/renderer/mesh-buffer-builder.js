'use strict';

export class MeshBufferBuilder {
    static build(meshData) {
        const { vertexCount, positions, covariances, opacities } = meshData;

        // vec3 pos + f opacity + 6f covariance + 2f _pad
        const floatsPerVertex = 12;
        const bufferData = new Float32Array(vertexCount * floatsPerVertex);

        // Layout per vertex:
        // [  px,   py,   pz, opacity,
        //   cxx,  cxy,  cxz,     ---,
        //   cyy,  cyz,  czz,     --- ]
        for (let i = 0; i < vertexCount; i++) {
            const baseSrc = i * 3;
            const baseCov = i * 6;
            const baseDst = i * floatsPerVertex;

            bufferData[baseDst + 0] = positions[baseSrc + 0];
            bufferData[baseDst + 1] = positions[baseSrc + 1];
            bufferData[baseDst + 2] = positions[baseSrc + 2];
            bufferData[baseDst + 3] = opacities[i];

            bufferData[baseDst + 4] = covariances[baseCov + 0];
            bufferData[baseDst + 5] = covariances[baseCov + 1];
            bufferData[baseDst + 6] = covariances[baseCov + 2];
            bufferData[baseDst + 7] = 0.0;

            bufferData[baseDst + 8]  = covariances[baseCov + 3];
            bufferData[baseDst + 9]  = covariances[baseCov + 4];
            bufferData[baseDst + 10] = covariances[baseCov + 5];
            bufferData[baseDst + 11] = 0.0;
        }

        return { vertexCount, floatsPerVertex, bufferData };
    }
}
