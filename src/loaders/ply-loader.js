'use strict';

const C0 = 0.28209;

export default class PLYLoader {
    constructor() {
        // hmm
    }

    expf(x) {
        return Math.exp(x);
    }

    sigmoid(x) {
        return 1 / (1 + this.expf(-x));
    }

    async load(file) {

        const buffer = await file.arrayBuffer();
        const textDecoder = new TextDecoder();
        const content = textDecoder.decode(buffer);

        // parse header
        const headerEnd = content.indexOf('end_header\n') + 'end_header\n'.length;
        const header = content.substring(0, headerEnd);

        let vertexCount = 0;
        const propertyOrder = [];

        for (const line of header.split('\n')) {
            if (line.startsWith('element vertex')) {
                // element vertex <number>
                vertexCount = parseInt(line.split(' ')[2]);
            } else if (line.startsWith('property')) {
                // property <type> <name>
                const parts = line.split(' ');
                propertyOrder.push(parts[2]); // store property name
            }
        }

        const positions = new Float32Array(vertexCount * 3);
        const scales = new Float32Array(vertexCount * 3);
        const rotations = new Float32Array(vertexCount * 4);
        const colors = new Float32Array(vertexCount * 3);
        const opacities = new Float32Array(vertexCount);

        const dataView = new DataView(buffer, headerEnd);
        const floatSize = 4; // float32
        const stride = propertyOrder.length * floatSize;

        // indices of the properties
        const idx = {
            x: propertyOrder.indexOf('x'),
            y: propertyOrder.indexOf('y'),
            z: propertyOrder.indexOf('z'),
            scale0: propertyOrder.indexOf('scale_0'),
            scale1: propertyOrder.indexOf('scale_1'),
            scale2: propertyOrder.indexOf('scale_2'),
            rot0: propertyOrder.indexOf('rot_0'),
            rot1: propertyOrder.indexOf('rot_1'),
            rot2: propertyOrder.indexOf('rot_2'),
            rot3: propertyOrder.indexOf('rot_3'),
            color0: propertyOrder.indexOf('f_dc_0'),
            color1: propertyOrder.indexOf('f_dc_1'),
            color2: propertyOrder.indexOf('f_dc_2'),
            opacity: propertyOrder.indexOf('opacity'),
        }

        // read binary data
        for (let i = 0; i < vertexCount; i++) {
            const baseOffset = i * stride;

            positions[i * 3 + 0] = dataView.getFloat32(baseOffset + idx.x * floatSize, true);
            positions[i * 3 + 1] = dataView.getFloat32(baseOffset + idx.y * floatSize, true);
            positions[i * 3 + 2] = dataView.getFloat32(baseOffset + idx.z * floatSize, true);

            scales[i * 3 + 0] = dataView.getFloat32(baseOffset + idx.scale0 * floatSize, true);
            scales[i * 3 + 1] = dataView.getFloat32(baseOffset + idx.scale1 * floatSize, true);
            scales[i * 3 + 2] = dataView.getFloat32(baseOffset + idx.scale2 * floatSize, true);

            // tf is this???
            // want (x,y,z,w) but stored as (w,x,y,z)
            rotations[i * 4 + 0] = dataView.getFloat32(baseOffset + idx.rot1 * floatSize, true);
            rotations[i * 4 + 1] = dataView.getFloat32(baseOffset + idx.rot2 * floatSize, true);
            rotations[i * 4 + 2] = dataView.getFloat32(baseOffset + idx.rot3 * floatSize, true);
            rotations[i * 4 + 3] = dataView.getFloat32(baseOffset + idx.rot0 * floatSize, true);
            
            colors[i * 3 + 0] = dataView.getFloat32(baseOffset + idx.color0 * floatSize, true);
            colors[i * 3 + 1] = dataView.getFloat32(baseOffset + idx.color1 * floatSize, true);
            colors[i * 3 + 2] = dataView.getFloat32(baseOffset + idx.color2 * floatSize, true);

            opacities[i] = dataView.getFloat32(baseOffset + idx.opacity * floatSize, true);
        }

        // normalize positions between -1 and 1
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        let minZ = Infinity, maxZ = -Infinity;
        for (let i = 0; i < vertexCount; i++) {
            const x = positions[i * 3 + 0];
            const y = positions[i * 3 + 1];
            const z = positions[i * 3 + 2];

            if (x < minX) minX = x; if (x > maxX) maxX = x;
            if (y < minY) minY = y; if (y > maxY) maxY = y;
            if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
        }
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        const centerZ = (minZ + maxZ) / 2;
        const rangeX = maxX - minX;
        const rangeY = maxY - minY;
        const rangeZ = maxZ - minZ;
        
        // scale factor: largest dimension fits in [-1,1]
        const maxRange = Math.max(rangeX, rangeY, rangeZ);
        const scale = 2 / maxRange;

        // apply centering and scaling, plus optional flip
        for (let i = 0; i < vertexCount; i++) {
            let x = positions[i * 3 + 0];
            let y = positions[i * 3 + 1];
            let z = positions[i * 3 + 2];

            // shift to center
            x -= centerX;
            y -= centerY;
            z -= centerZ;

            // scale
            x *= scale;
            y *= scale;
            z *= scale;

            // apply expf and scale
            scales[i * 3 + 0] = this.expf(scales[i * 3 + 0]) * scale;
            scales[i * 3 + 1] = this.expf(scales[i * 3 + 1]) * scale;
            scales[i * 3 + 2] = this.expf(scales[i * 3 + 2]) * scale;

            // y and z flip
            positions[i * 3 + 0] = x;
            positions[i * 3 + 1] = y;
            positions[i * 3 + 2] = z;

            // apply sigmoid to opacities
            opacities[i] = this.sigmoid(opacities[i]);

            // normalize colors to [0,1]
            colors[i * 3 + 0] = colors[i * 3 + 0] * C0 + 0.5;
            colors[i * 3 + 1] = colors[i * 3 + 1] * C0 + 0.5;
            colors[i * 3 + 2] = colors[i * 3 + 2] * C0 + 0.5;
        }

        return {
            vertexCount,
            positions,
            scales,
            rotations,
            colors,
            opacities
        };
    }
}