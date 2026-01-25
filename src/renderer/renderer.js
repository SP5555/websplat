'use strict';

import Camera from "./camera.js";
import WGSLShader from "./wgsl-shader/wgsl-shader.js";

export default class Renderer {
    constructor(input) {
        this.canvas = document.getElementById('canvas00');
        this.device = null;
        this.context = null;

        this.vertexBuffer = null;
        this.vertexCount = 0;
        this.pipeline = null;

        this.camera = new Camera(input, this.canvas.width / this.canvas.height);
        this.cameraBuffer = null;
        this.cameraBindGroup = null;

        this.init();
    }

    async init() {
        this.resizeCanvas();
        this.initializeEventListeners();

        if (!(await this.initDevice())) {
            console.error("Failed to initialize WebGPU device.");
            return;
        }

        this.createCameraBuffer();
        await this.createPipeline();
    }

    /* ===== ===== GPU Setup ===== ===== */

    async initDevice() {
        if (!navigator.gpu) {
            console.error("WebGPU not supported in this browser.");
            return false;
        }

        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) {
            console.error("Failed to get GPU adapter.");
            return false;
        }

        this.device = await adapter.requestDevice();
        this.context = this.canvas.getContext('webgpu');
        this.configureContext();

        return true;
    }

    createCameraBuffer() {
        this.cameraBuffer = this.device.createBuffer({
            label: "Camera Buffer",
            size: 2 * 16 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
    }

    async createPipeline() {
        const shader = new WGSLShader(this.device, './assets/shader.wgsl');
        await shader.load();

        this.pipeline = this.device.createRenderPipeline({
            label: "Render Pipeline",
            layout: 'auto',
            vertex: {
                module: shader.getModule(),
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 3 * 4,
                    attributes: [{ format: 'float32x3', offset: 0, shaderLocation: 0 }]
                }]
            },
            fragment: {
                module: shader.getModule(),
                entryPoint: 'fs_main',
                targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }]
            },
            primitive: { topology: 'point-list' }
        });

        this.cameraBindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: this.cameraBuffer } }]
        });

        this.vertexBuffer = this.device.createBuffer({
            label: "Vertex Buffer",
            size: 0,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
    }

    configureContext() {
        if (!this.device || !this.context) return;
        this.context.configure({
            device: this.device,
            format: navigator.gpu.getPreferredCanvasFormat(),
            alphaMode: 'opaque'
        });
    }

    /* ===== ===== Event Handling ===== ===== */

    initializeEventListeners() {
        window.addEventListener('resize', () => this.onWindowResize(), false);
    }

    onWindowResize() {
        this.resizeCanvas();
        this.configureContext();
        this.camera.updateAspect(this.canvas.width / this.canvas.height);
    }

    resizeCanvas() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    /* ===== ===== Mesh Management ===== ===== */

    setMeshData(meshData) {
        const positions = meshData.positions;
        this.vertexCount = positions.length / 3;

        this.reallocateVertexBuffer(positions.byteLength);
        this.device.queue.writeBuffer(this.vertexBuffer, 0, positions);
    }

    reallocateVertexBuffer(size) {
        if (this.vertexBuffer) this.vertexBuffer.destroy();
        this.vertexBuffer = this.device.createBuffer({
            label: "Vertex Buffer",
            size: size,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
    }

    /* ===== ===== Rendering ===== ===== */

    render(dt) {
        if (!this.device || !this.context || !this.pipeline) return;

        this.updateCameraBuffer(dt);

        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: 'clear',
                storeOp: 'store'
            }]
        });

        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.cameraBindGroup);
        pass.setVertexBuffer(0, this.vertexBuffer);
        if (this.vertexCount) pass.draw(this.vertexCount, 1);

        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }

    updateCameraBuffer(dt) {
        this.camera.update(dt);

        const cameraData = new Float32Array(32);
        cameraData.set(this.camera.vMatrix, 0);
        cameraData.set(this.camera.pMatrix, 16);

        this.device.queue.writeBuffer(this.cameraBuffer, 0, cameraData.buffer);
    }
}
