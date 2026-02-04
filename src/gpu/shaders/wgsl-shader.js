'use strict';

export default class WGSLShader {
    constructor(device, path) {
        this.device = device;
        this.path = path;     // path to WGSL file
        this.module = null;   // GPUShaderModule after loading
    }

    // Load the WGSL file and create a GPUShaderModule
    async load() {
        try {
            const response = await fetch(this.path);
            if (!response.ok) throw new Error(`Failed to load shader: ${this.path}`);

            const code = await response.text();
            this.module = this.device.createShaderModule({
                label: 'WGSL Shader Module',
                code: code,
            });

            return this.module;
        } catch (err) {
            console.error(err);
            return null;
        }
    }

    getModule() {
        if (!this.module) {
            console.warn('Shader module not loaded yet!');
        }
        return this.module;
    }
}
