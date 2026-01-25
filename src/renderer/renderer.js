'use strict';

import Camera from "./camera.js";
import WGSLShader from "./wgsl-shader/wgsl-shader.js";

export default class Renderer {
    constructor(input) {
        this.device = null;
        this.context = null;
        this.canvas = document.getElementById('canvas00');

        this.vertexCount = null;
        this.vertexBuffer = null;
        this.pipeline = null;

        this.camera = new Camera(input, this.canvas.width / this.canvas.height);
        this.cameraBuffer = null;

        this.initializeRenderer();
        this.initializeEventListeners();
    }

    async initializeRenderer() {
        this.resizeCanvas();

        if (!navigator.gpu) {
            console.error("WebGPU not supported in this browser.");
            return;
        }

        // Request adapter + device
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) {
            console.error("Failed to get GPU adapter.");
            return;
        }
        this.device = await adapter.requestDevice();

        this.context = this.canvas.getContext('webgpu');
        this.configureContext();

        // camera
        this.cameraBuffer = this.device.createBuffer({
            label: "Camera Buffer",
            size: 2 * 16 * 4, // two 4x4 f32 matrices
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // load shader
        const shader = new WGSLShader(this.device, './assets/shader.wgsl');
        await shader.load();
        const shaderModule = shader.getModule();

        this.vertexBuffer = this.device.createBuffer({
            label: "Vertex Buffer",
            size: 0,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });

        this.pipeline = this.device.createRenderPipeline({
            label: "Render Pipeline",
            layout: 'auto',
            vertex: {
                module: shaderModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 3 * 4,
                    attributes: [{
                        format: 'float32x3',
                        offset: 0,
                        shaderLocation: 0,
                    }]
                }]
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: navigator.gpu.getPreferredCanvasFormat()
                }]
            },
            primitive: { topology: 'point-list' }
        });

        this.cameraBindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [{
                binding: 0,
                resource: {
                    buffer: this.cameraBuffer
                }
            }]
        });
    }

    initializeEventListeners() {
        window.addEventListener('resize', () => this.onWindowResize(), false);
    }

    setMeshData(meshData) {
        const positions = meshData.positions;
        this.vertexCount = positions.length / 3;

        // Reallocate GPU buffer if needed
        this.vertexBuffer.destroy();
        this.vertexBuffer = this.device.createBuffer({
            label: "Vertex Buffer",
            size: positions.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });

        this.device.queue.writeBuffer(this.vertexBuffer, 0, positions);
    }

    resizeCanvas() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    configureContext() {
        if (!this.device || !this.context) return;

        this.context.configure({
            device: this.device,
            format: navigator.gpu.getPreferredCanvasFormat(),
            alphaMode: 'opaque'
        });
    }

    render(dt) {
        if ( !this.device || !this.context || !this.pipeline ) return;

        this.camera.update(dt);
        const cameraData = new Float32Array(32);
        cameraData.set(this.camera.vMatrix, 0);
        cameraData.set(this.camera.pMatrix, 16);
        this.device.queue.writeBuffer(this.cameraBuffer, 0, cameraData.buffer);

        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store'
            }]
        });

        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.cameraBindGroup);
        pass.setVertexBuffer(0, this.vertexBuffer);
        if (this.vertexCount) {
            pass.draw(this.vertexCount, 1, 0, 0);
        }
        pass.end();

        this.device.queue.submit([encoder.finish()]);
    }

    onWindowResize() {
        this.resizeCanvas();
        this.configureContext();
        this.camera.updateAspect(this.canvas.width / this.canvas.height);
    }
}
