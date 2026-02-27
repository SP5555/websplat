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

        for (let i = 0; i < vertexCount; i++) {
            scales[i * 3 + 0] = this.expf(scales[i * 3 + 0]);
            scales[i * 3 + 1] = this.expf(scales[i * 3 + 1]);
            scales[i * 3 + 2] = this.expf(scales[i * 3 + 2]);

            // apply sigmoid to opacities
            opacities[i] = this.sigmoid(opacities[i]);

            // normalize colors to [0,1]
            colors[i * 3 + 0] = colors[i * 3 + 0] * C0 + 0.5;
            colors[i * 3 + 1] = colors[i * 3 + 1] * C0 + 0.5;
            colors[i * 3 + 2] = colors[i * 3 + 2] * C0 + 0.5;
        }

        console.log(`Loaded PLY: ${vertexCount} vertices`);

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