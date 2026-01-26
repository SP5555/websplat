'use strict';

import Camera from "./camera.js";
import { MeshBufferBuilder } from "./mesh-buffer-builder.js";
import WGSLShader from "./wgsl-shader/wgsl-shader.js";

export default class Renderer {
    constructor(input) {
        this.canvas = document.getElementById('canvas00');
        this.device = null;
        this.context = null;

        this.vertexBuffer = null;
        this.vertexCount = 0;
        this.floatsPerVertex = 0;
        this.finalRenderPipeline = null;

        this.camera = new Camera(input, this.canvas.width / this.canvas.height);
        this.cameraBuffer = null;
        this.cameraBindGroup = null;

        this.GRID_SIZE = { x: 32, y: 16 };
        this.MAX_VERTICES_PER_TILE = 2048;

        this.init();
    }

    async init() {
        this.resizeCanvas();
        this.initializeEventListeners();

        if (!(await this.initDevice())) {
            console.error("Failed to initialize WebGPU device.");
            return;
        }

        await this.initGPUPipeline();
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

    async initGPUPipeline() {
        this.createCameraBuffer();
        await this.createTransformPipeline();
        await this.createTilePipeline();
        // await this.createSortPipeline();
        await this.createFinalRenderPipeline();
    }

    createCameraBuffer() {
        this.cameraBuffer = this.device.createBuffer({
            label: "Camera Buffer",
            size: 2 * 16 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
    }

    async createTransformPipeline() {
        const shader = new WGSLShader(this.device, './shaders/transform.wgsl');
        await shader.load();

        this.transformInputBuffer = this.device.createBuffer({
            label: "Transform Input Buffer",
            size: 80, // minimum size
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });

        this.transformOutputBuffer = this.device.createBuffer({
            label: "Transform Output Buffer",
            size: 80, // minimum size
            usage: GPUBufferUsage.STORAGE
        });

        this.transformPipeline = this.device.createComputePipeline({
            label: "Transform Pipeline",
            layout: 'auto',
            compute: {
                module: shader.getModule(),
                entryPoint: 'cs_main'
            }
        });

        this.transformBindGroup = this.device.createBindGroup({
            layout: this.transformPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.cameraBuffer } },
                { binding: 1, resource: { buffer: this.transformInputBuffer } },
                { binding: 2, resource: { buffer: this.transformOutputBuffer } }
            ]
        });
    }

    async createTilePipeline() {
        const shader = new WGSLShader(this.device, './shaders/tile.wgsl');
        await shader.load();

        this.tileVerticesBuffer = this.device.createBuffer({
            label: "Tile Vertices Buffer",
            size: Math.max(80, this.GRID_SIZE.x * this.GRID_SIZE.y * this.MAX_VERTICES_PER_TILE * this.floatsPerVertex * 4),
            usage: GPUBufferUsage.STORAGE
        });

        // bin counter is reset each frame, requires COPY_DST
        this.tileCountersBuffer = this.device.createBuffer({
            label: "Tile Counters Buffer",
            size: this.GRID_SIZE.x * this.GRID_SIZE.y * 4, // 1x uint32 per tile
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });

        // vertex count in the scene, grid sizes and max vertices a tile can hold
        // COPY_DST for future GRID_SIZE or MAX_VERTICES_PER_TILE changes
        this.tileParamsBuffer = this.device.createBuffer({
            label: "Tile Params Buffer",
            size: 4 * 4, // 4x uint32
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });

        const tileParams = new Uint32Array([this.vertexCount, this.GRID_SIZE.x, this.GRID_SIZE.y, this.MAX_VERTICES_PER_TILE]);
        this.device.queue.writeBuffer(this.tileParamsBuffer, 0, tileParams.buffer);

        this.tilePipeline = this.device.createComputePipeline({
            label: "Tile Pipeline",
            layout: 'auto',
            compute: {
                module: shader.getModule(),
                entryPoint: 'cs_main'
            }
        });

        this.tileBindGroup = this.device.createBindGroup({
            layout: this.tilePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.transformOutputBuffer } },
                { binding: 1, resource: { buffer: this.tileVerticesBuffer } },
                { binding: 2, resource: { buffer: this.tileCountersBuffer } },
                { binding: 3, resource: { buffer: this.tileParamsBuffer } }
            ]
        });
    }

    async createSortPipeline() {
        const shader = new WGSLShader(this.device, './shaders/sort.wgsl');
        await shader.load();

        this.sortPipeline = this.device.createComputePipeline({
            label: "Sort Pipeline",
            layout: 'auto',
            compute: {
                module: shader.getModule(),
                entryPoint: 'cs_main'
            }
        });

        this.sortBindGroup = this.device.createBindGroup({
            layout: this.sortPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.tileVerticesBuffer } },
                { binding: 1, resource: { buffer: this.tileCountersBuffer } },
                { binding: 2, resource: { buffer: this.tileParamsBuffer } }
            ]
        });
    }

    // async createFinalRenderPipeline() {
    //     const shader = new WGSLShader(this.device, './shaders/basic-shader.wgsl');
    //     await shader.load();

    //     this.finalRenderPipeline = this.device.createRenderPipeline({
    //         label: "Render Pipeline",
    //         layout: 'auto',
    //         vertex: {
    //             module: shader.getModule(),
    //             entryPoint: 'vs_main',
    //             buffers: [{
    //                 arrayStride: this.floatsPerVertex * 4,
    //                 attributes: [
    //                     { shaderLocation: 0, format: 'float32x4', offset: 0 },     // position
    //                     { shaderLocation: 1, format: 'float32', offset: 3 * 4 },   // opacity
    //                     { shaderLocation: 2, format: 'float32x4', offset: 4 * 4 }, // covariance part 1
    //                     { shaderLocation: 3, format: 'float32x4', offset: 8 * 4 }, // covariance part 2
    //                     { shaderLocation: 4, format: 'float32x3', offset: 12 * 4 } // color
    //                 ]
    //             }]
    //         },
    //         fragment: {
    //             module: shader.getModule(),
    //             entryPoint: 'fs_main',
    //             targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }]
    //         },
    //         primitive: { topology: 'point-list' }
    //     });

    //     this.cameraBindGroup = this.device.createBindGroup({
    //         layout: this.finalRenderPipeline.getBindGroupLayout(0),
    //         entries: [{ binding: 0, resource: { buffer: this.cameraBuffer } }]
    //     });

    //     this.vertexBuffer = this.device.createBuffer({
    //         label: "Vertex Buffer",
    //         size: 0,
    //         usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    //     });
    // }

    async createFinalRenderPipeline() {
        const shader = new WGSLShader(this.device, './shaders/final-render.wgsl');
        await shader.load();

        this.finalRenderPipeline = this.device.createRenderPipeline({
            label: "Final Render Pipeline",
            layout: 'auto',
            vertex: {
                module: shader.getModule(),
                entryPoint: 'vs_main',
                buffers: [] // fullscreen triangle
            },
            fragment: {
                module: shader.getModule(),
                entryPoint: 'fs_main',
                targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }]
            },
            primitive: { topology: 'triangle-list' }
        });

        // COPY_DST for canvas resize updates
        this.canvasParamsBuffer = this.device.createBuffer({
            label: "Canvas Params Buffer",
            size: 2 * 4, // width and height as uint32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        const canvasParams = new Uint32Array([this.canvas.width, this.canvas.height]);
        this.device.queue.writeBuffer(this.canvasParamsBuffer, 0, canvasParams.buffer);

        this.finalRenderBindGroup = this.device.createBindGroup({
            layout: this.finalRenderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.tileVerticesBuffer } },
                { binding: 1, resource: { buffer: this.tileCountersBuffer } },
                { binding: 2, resource: { buffer: this.tileParamsBuffer } },
                { binding: 3, resource: { buffer: this.canvasParamsBuffer } }
            ]
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

        this.device?.queue.writeBuffer(
            this.canvasParamsBuffer,
            0,
            new Uint32Array([this.canvas.width, this.canvas.height]).buffer
        );
    }

    /* ===== ===== Mesh Management ===== ===== */

    setMeshData(meshData) {
        const { vertexCount, floatsPerVertex, bufferData } = MeshBufferBuilder.build(meshData);
        this.vertexCount = vertexCount;
        this.floatsPerVertex = floatsPerVertex;

        this.reallocateVertexBuffer(bufferData);

        const tileParams = new Uint32Array([
            vertexCount,
            this.GRID_SIZE.x,
            this.GRID_SIZE.y,
            this.MAX_VERTICES_PER_TILE
        ]);

        // this.device.queue.writeBuffer(this.vertexBuffer, 0, bufferData.buffer);
        this.device.queue.writeBuffer(this.transformInputBuffer, 0, bufferData.buffer);
        this.device.queue.writeBuffer(this.tileParamsBuffer, 0, tileParams.buffer);
    }

    reallocateVertexBuffer(bufferData) {
        // if (this.vertexBuffer) this.vertexBuffer.destroy();
        // this.vertexBuffer = this.device.createBuffer({
        //     label: "Vertex Buffer",
        //     size: bufferData.byteLength,
        //     usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        // });

        if (this.transformInputBuffer) this.transformInputBuffer.destroy();
        this.transformInputBuffer = this.device.createBuffer({
            label: "Transform Input Buffer",
            size: bufferData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });

        if (this.transformOutputBuffer) this.transformOutputBuffer.destroy();
        this.transformOutputBuffer = this.device.createBuffer({
            label: "Transform Output Buffer",
            size: bufferData.byteLength,
            usage: GPUBufferUsage.STORAGE
        });

        if (this.tileVerticesBuffer) this.tileVerticesBuffer.destroy();
        this.tileVerticesBuffer = this.device.createBuffer({
            label: "Tile Vertices Buffer",
            size: Math.max(80, this.GRID_SIZE.x * this.GRID_SIZE.y * this.MAX_VERTICES_PER_TILE * this.floatsPerVertex * 4),
            usage: GPUBufferUsage.STORAGE
        });

        this.transformBindGroup = this.device.createBindGroup({
            layout: this.transformPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.cameraBuffer } },
                { binding: 1, resource: { buffer: this.transformInputBuffer } },
                { binding: 2, resource: { buffer: this.transformOutputBuffer } }
            ]
        });

        this.tileBindGroup = this.device.createBindGroup({
            layout: this.tilePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.transformOutputBuffer } },
                { binding: 1, resource: { buffer: this.tileVerticesBuffer } },
                { binding: 2, resource: { buffer: this.tileCountersBuffer } },
                { binding: 3, resource: { buffer: this.tileParamsBuffer } }
            ]
        });

        // this.sortBindGroup = this.device.createBindGroup({
        //     layout: this.sortPipeline.getBindGroupLayout(0),
        //     entries: [
        //         { binding: 0, resource: { buffer: this.tileVerticesBuffer } },
        //         { binding: 1, resource: { buffer: this.tileCountersBuffer } },
        //         { binding: 2, resource: { buffer: this.tileParamsBuffer } }
        //     ]
        // });

        this.finalRenderBindGroup = this.device.createBindGroup({
            layout: this.finalRenderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.tileVerticesBuffer } },
                { binding: 1, resource: { buffer: this.tileCountersBuffer } },
                { binding: 2, resource: { buffer: this.tileParamsBuffer } },
                { binding: 3, resource: { buffer: this.canvasParamsBuffer } }
            ]
        });
    }
    /* ===== ===== Rendering ===== ===== */

    async render(dt) {
        if (!this.device || !this.context || !this.finalRenderPipeline) return;

        this.updateCameraBuffer(dt);

        this.device.queue.writeBuffer(this.tileCountersBuffer, 0, new Uint32Array(this.tileCountersBuffer.size / 4).fill(0).buffer);
        
        const encoder = this.device.createCommandEncoder();

        // Pass 1: Transform Compute Pass
        {
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.transformPipeline);
            pass.setBindGroup(0, this.transformBindGroup);
            const numWorkgroups = Math.max(8, Math.ceil(this.vertexCount / 64));
            pass.dispatchWorkgroups(numWorkgroups);
            pass.end();
        }

        // Pass 2: Binning Pass
        {
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.tilePipeline);
            pass.setBindGroup(0, this.tileBindGroup);
            const numWorkgroups = Math.max(8, Math.ceil(this.vertexCount / 64));
            pass.dispatchWorkgroups(numWorkgroups);
            pass.end();
        }

        // Pass 3: Sorting Pass
        // {

        //     const pass = encoder.beginComputePass();
        //     pass.setPipeline(this.sortPipeline);
        //     pass.setBindGroup(0, this.sortBindGroup);
        //     pass.dispatchWorkgroups(this.GRID_SIZE.x * this.GRID_SIZE.y);
        //     pass.end();

        // }

        // Pass 4: Final Render Pass
        {
            const pass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: this.context.getCurrentTexture().createView(),
                    clearValue: { r: 0, g: 0, b: 0, a: 1 },
                    loadOp: 'clear',
                    storeOp: 'store'
                }]
            });
            pass.setPipeline(this.finalRenderPipeline);
            // pass.setBindGroup(0, this.cameraBindGroup);
            // pass.setVertexBuffer(0, this.vertexBuffer);
            // if (this.vertexCount) pass.draw(this.vertexCount, 1);
            pass.setBindGroup(0, this.finalRenderBindGroup);
            pass.draw(3); // fullscreen triangle
            pass.end();
        }
        
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
