'use strict';

import Camera from "../scene/camera.js";
import { SplatDataPacker } from "../gaussian/splat-data-packer.js";
import WGSLShader from "../gpu/shaders/wgsl-shader.js";
import { eventBus } from "../utils/event-emitters.js";
import { EVENTS } from "../utils/event.js";

const BUFFER_MIN_SIZE = 80; // bytes

export default class ComputeSplatRenderer {
    constructor(input) {
        // max buffer size limit = 2^27 bytes
        // GRID_SIZE.x * GRID_SIZE.y * MAX_SPLATS_PER_TILE * 4 <= 2^27
        // keep the grid dimensions the power of 2
        // otherwise, sorting shader won't work correctly
        this.GRID_SIZE = { x: 32, y: 32 };
        this.MAX_SPLATS_PER_TILE = Math.pow(2, 25 - Math.log2(this.GRID_SIZE.x * this.GRID_SIZE.y));

        this.clearTilesShaderPath = './src/gpu/shaders/compute/clear-tiles/clear-tiles.wgsl';
        this.transformShaderPath  = './src/gpu/shaders/compute/transform/transform.wgsl';
        this.tileShaderPath       = './src/gpu/shaders/compute/tile/tile.wgsl';
        this.sortShaderPath       = './src/gpu/shaders/compute/sort/sort-bitonic.wgsl';
        this.renderShaderPath     = './src/gpu/shaders/compute/render/render.wgsl';

        /* ===== Private Zone ===== */
        this.canvas = document.getElementById('canvas00');
        this.device = null;
        this.context = null;
        
        this.camera = new Camera(input, this.canvas.width / this.canvas.height);
        this.splatCount = 0;
        this.FLOATS_PER_SPLAT3D = 16;
        this.FLOATS_PER_SPLAT2D = 12;

        /* ===== Render Params ===== */
        this.scaleMultiplier = 1.0;
        this.showSfMPoints = 0.0; // 0.0 = false, 1.0 = true
        
        /* ===== Pipelines ===== */
        // 0th "reset" pipeline
        this.clearTilesPipeline = null;
        // main 4 stages
        this.transformPipeline = null;
        this.tilePipeline = null;
        this.sortPipeline = null;
        this.renderPipeline = null;
        
        this.isPipelineInitialized = false;

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

        console.log("Device limits: ", this.device.limits);
        return true;
    }

    configureContext() {
        if (!this.device || !this.context) return;
        this.context.configure({
            device: this.device,
            format: navigator.gpu.getPreferredCanvasFormat(),
            alphaMode: 'opaque'
        });
    }

    async initGPUPipeline() {
        await this.initShaders();

        this.createDataBuffers();

        this.createClearTilesPipeline();
        this.createTransformPipeline();
        this.createTilePipeline();
        this.createSortPipeline();
        this.createRenderPipeline();

        this.isPipelineInitialized = true;
    }

    async initShaders() {
        this.clearTileShader = new WGSLShader(this.device, this.clearTilesShaderPath);
        await this.clearTileShader.load();

        this.transformShader = new WGSLShader(this.device, this.transformShaderPath);
        await this.transformShader.load();
        
        this.tileShader = new WGSLShader(this.device, this.tileShaderPath);
        await this.tileShader.load();

        this.sortShader = new WGSLShader(this.device, this.sortShaderPath);
        await this.sortShader.load();

        this.renderShader = new WGSLShader(this.device, this.renderShaderPath);
        await this.renderShader.load();
    }

    createDataBuffers() {
        // holds three camera matrices: view, projection, projection*view
        this.cameraBuffer = this.device.createBuffer({
            label: "Camera Buffer",
            size: 3 * 16 * 4, // 3x mat4x4<f32>
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // holds GUI-controlled render parameters
        this.renderParamsBuffer = this.device.createBuffer({
            label: "Render Params Buffer",
            size: 8, // 2x f32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.updateRenderParams();

        // buffer that holds the input splat data for the transform pass
        this.spat3DBuffer = this.device.createBuffer({
            label: "Splat 3D Buffer",
            size: BUFFER_MIN_SIZE,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });

        // buffer that holds the transformed splat data from the transform pass
        this.spat2DBuffer = this.device.createBuffer({
            label: "Splat 2D Buffer",
            size: BUFFER_MIN_SIZE,
            usage: GPUBufferUsage.STORAGE
        });

        // buffer that holds the transformed splat Z positions,
        // used for depth sorting during the sort pass for faster memory access
        this.depthKeysBuffer = this.device.createBuffer({
            label: "Depth Keys Buffer",
            size: BUFFER_MIN_SIZE,
            usage: GPUBufferUsage.STORAGE
        });

        // each tile holds MAX_SPLATS_PER_TILE indices (uint32)
        this.tileIndicesBuffer = this.device.createBuffer({
            label: "Tile Indices Buffer",
            size: Math.max(BUFFER_MIN_SIZE, this.GRID_SIZE.x * this.GRID_SIZE.y * this.MAX_SPLATS_PER_TILE * 4),
            usage: GPUBufferUsage.STORAGE
        });

        // holds the current number of splats in each tile (uint32)
        this.tileCountersBuffer = this.device.createBuffer({
            label: "Tile Counters Buffer",
            size: this.GRID_SIZE.x * this.GRID_SIZE.y * 4, // 1x uint32 per tile
            usage: GPUBufferUsage.STORAGE
        });

        // splat count in the scene, grid sizes and max splats a tile can hold
        // COPY_DST for future GRID_SIZE or MAX_SPLATS_PER_TILE changes
        this.globalParamsBuffer = this.device.createBuffer({
            label: "Global Params Buffer",
            size: 4 * 4, // 4x uint32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        const globalParams = new Uint32Array([this.splatCount, this.GRID_SIZE.x, this.GRID_SIZE.y, this.MAX_SPLATS_PER_TILE]);
        this.device.queue.writeBuffer(this.globalParamsBuffer, 0, globalParams.buffer);

        // canvas width and height
        // COPY_DST for canvas resize updates
        this.canvasParamsBuffer = this.device.createBuffer({
            label: "Canvas Params Buffer",
            size: 2 * 4, // width and height as uint32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        const canvasParams = new Uint32Array([this.canvas.width, this.canvas.height]);
        this.device.queue.writeBuffer(this.canvasParamsBuffer, 0, canvasParams.buffer);
    }

    createClearTilesPipeline() {
        this.clearTilesPipeline = this.device.createComputePipeline({
            label: "Clear Tiles Pipeline",
            layout: 'auto',
            compute: {
                module: this.clearTileShader.getModule(),
                entryPoint: 'cs_main'
            }
        });

        this.clearTilesBindGroup0 = this.device.createBindGroup({
            layout: this.clearTilesPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.globalParamsBuffer } }
            ]
        });

        this.clearTilesBindGroup1 = this.device.createBindGroup({
            layout: this.clearTilesPipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.tileIndicesBuffer } },
                { binding: 1, resource: { buffer: this.tileCountersBuffer } }
            ]
        });
    }

    createTransformPipeline() {
        this.transformPipeline = this.device.createComputePipeline({
            label: "Transform Pipeline",
            layout: 'auto',
            compute: {
                module: this.transformShader.getModule(),
                entryPoint: 'cs_main'
            }
        });

        this.transformBindGroup0 = this.device.createBindGroup({
            layout: this.transformPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.cameraBuffer } },
                { binding: 1, resource: { buffer: this.canvasParamsBuffer } },
                { binding: 2, resource: { buffer: this.renderParamsBuffer } }
            ]
        });

        this.transformBindGroup1 = this.device.createBindGroup({
            layout: this.transformPipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.spat3DBuffer } },
                { binding: 1, resource: { buffer: this.spat2DBuffer } },
                { binding: 2, resource: { buffer: this.depthKeysBuffer } }
            ]
        });
    }

    createTilePipeline() {
        this.tilePipeline = this.device.createComputePipeline({
            label: "Tile Pipeline",
            layout: 'auto',
            compute: {
                module: this.tileShader.getModule(),
                entryPoint: 'cs_main'
            }
        });

        this.tileBindGroup0 = this.device.createBindGroup({
            layout: this.tilePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.globalParamsBuffer } },
                { binding: 1, resource: { buffer: this.canvasParamsBuffer } }
            ]
        });

        this.tileBindGroup1 = this.device.createBindGroup({
            layout: this.tilePipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.spat2DBuffer } },
                { binding: 1, resource: { buffer: this.tileIndicesBuffer } },
                { binding: 2, resource: { buffer: this.tileCountersBuffer } }
            ]
        });
    }

    createSortPipeline() {
        this.sortPipeline = this.device.createComputePipeline({
            label: "Sort Pipeline",
            layout: 'auto',
            compute: {
                module: this.sortShader.getModule(),
                entryPoint: 'cs_main'
            }
        });

        this.sortBindGroup0 = this.device.createBindGroup({
            layout: this.sortPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.globalParamsBuffer } }
            ]
        });
        
        this.sortBindGroup1 = this.device.createBindGroup({
            layout: this.sortPipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.depthKeysBuffer } },
                { binding: 1, resource: { buffer: this.tileIndicesBuffer } },
                { binding: 2, resource: { buffer: this.tileCountersBuffer } }
            ]
        });
    }

    createRenderPipeline() {
        this.renderPipeline = this.device.createRenderPipeline({
            label: "Render Pipeline",
            layout: 'auto',
            vertex: {
                module: this.renderShader.getModule(),
                entryPoint: 'vs_main',
                buffers: [] // fullscreen triangle
            },
            fragment: {
                module: this.renderShader.getModule(),
                entryPoint: 'fs_main',
                targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }]
            },
            primitive: { topology: 'triangle-list' }
        });

        this.renderBindGroup0 = this.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.globalParamsBuffer } },
                { binding: 1, resource: { buffer: this.canvasParamsBuffer } }
            ]
        });

        this.renderBindGroup1 = this.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.spat2DBuffer } },
                { binding: 1, resource: { buffer: this.tileIndicesBuffer } },
                { binding: 2, resource: { buffer: this.tileCountersBuffer } }
            ]
        });
    }

    /* ===== ===== Event Handling ===== ===== */

    initializeEventListeners() {
        window.addEventListener('resize', () => this.onWindowResize(), false);

        eventBus.on(EVENTS.SCALE_MULTIPLIER_CHANGE, value => {
            this.scaleMultiplier = value;
            this.updateRenderParams();
        });

        eventBus.on(EVENTS.SHOW_SFM_CHANGE, value => {
            this.showSfMPoints = value ? 1.0 : 0.0;
            this.updateRenderParams();
        });
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
        const { splatCount, bufferData } = SplatDataPacker.build(meshData);
        this.splatCount = splatCount;

        this.reallocateBuffers(splatCount);

        this.device.queue.writeBuffer(this.spat3DBuffer, 0, bufferData.buffer);

        const globalParams = new Uint32Array([
            splatCount,
            this.GRID_SIZE.x,
            this.GRID_SIZE.y,
            this.MAX_SPLATS_PER_TILE
        ]);
        this.device.queue.writeBuffer(this.globalParamsBuffer, 0, globalParams.buffer);
    }

    reallocateBuffers(splatCount) {
        if (this.spat3DBuffer) this.spat3DBuffer.destroy();
        this.spat3DBuffer = this.device.createBuffer({
            label: "Splat 3D Buffer",
            size: splatCount * this.FLOATS_PER_SPLAT3D * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });

        if (this.spat2DBuffer) this.spat2DBuffer.destroy();
        this.spat2DBuffer = this.device.createBuffer({
            label: "Splat 2D Buffer",
            size: splatCount * this.FLOATS_PER_SPLAT2D * 4,
            usage: GPUBufferUsage.STORAGE
        });

        if (this.depthKeysBuffer) this.depthKeysBuffer.destroy();
        this.depthKeysBuffer = this.device.createBuffer({
            label: "Depth Keys Buffer",
            size: splatCount * 4, // 1x f32 per splat
            usage: GPUBufferUsage.STORAGE
        });

        if (this.tileIndicesBuffer) this.tileIndicesBuffer.destroy();
        this.tileIndicesBuffer = this.device.createBuffer({
            label: "Tile Indices Buffer",
            size: Math.max(BUFFER_MIN_SIZE, this.GRID_SIZE.x * this.GRID_SIZE.y * this.MAX_SPLATS_PER_TILE * 4),
            usage: GPUBufferUsage.STORAGE
        });

        this.clearTilesBindGroup1 = this.device.createBindGroup({
            layout: this.clearTilesPipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.tileIndicesBuffer } },
                { binding: 1, resource: { buffer: this.tileCountersBuffer } }
            ]
        });

        this.transformBindGroup1 = this.device.createBindGroup({
            layout: this.transformPipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.spat3DBuffer } },
                { binding: 1, resource: { buffer: this.spat2DBuffer } },
                { binding: 2, resource: { buffer: this.depthKeysBuffer } }
            ]
        });

        this.tileBindGroup1 = this.device.createBindGroup({
            layout: this.tilePipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.spat2DBuffer } },
                { binding: 1, resource: { buffer: this.tileIndicesBuffer } },
                { binding: 2, resource: { buffer: this.tileCountersBuffer } }
            ]
        });

        this.sortBindGroup1 = this.device.createBindGroup({
            layout: this.sortPipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.depthKeysBuffer } },
                { binding: 1, resource: { buffer: this.tileIndicesBuffer } },
                { binding: 2, resource: { buffer: this.tileCountersBuffer } }
            ]
        });

        this.renderBindGroup1 = this.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.spat2DBuffer } },
                { binding: 1, resource: { buffer: this.tileIndicesBuffer } },
                { binding: 2, resource: { buffer: this.tileCountersBuffer } }
            ]
        });
    }

    /* ===== ===== Rendering ===== ===== */

    async render(dt) {
        if (!this.device || !this.context || !this.isPipelineInitialized) return;

        this.updateCameraBuffer(dt);

        const encoder = this.device.createCommandEncoder();

        // Pass 0: Clear Tiles Pass
        {
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.clearTilesPipeline);
            pass.setBindGroup(0, this.clearTilesBindGroup0);
            pass.setBindGroup(1, this.clearTilesBindGroup1);
            const numWorkgroups = this.GRID_SIZE.x * this.GRID_SIZE.y;
            pass.dispatchWorkgroups(numWorkgroups);
            pass.end();
        }

        // Pass 1: Transform Compute Pass
        {
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.transformPipeline);
            pass.setBindGroup(0, this.transformBindGroup0);
            pass.setBindGroup(1, this.transformBindGroup1);
            const WGSize = 128;
            const numWorkgroups = Math.max(8, Math.ceil(this.splatCount / WGSize));
            pass.dispatchWorkgroups(numWorkgroups);
            pass.end();
        }

        // Pass 2: Tiling Pass
        {
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.tilePipeline);
            pass.setBindGroup(0, this.tileBindGroup0);
            pass.setBindGroup(1, this.tileBindGroup1);
            const WGSize = 128;
            const numWorkgroups = Math.max(8, Math.ceil(this.splatCount / WGSize));
            pass.dispatchWorkgroups(numWorkgroups);
            pass.end();
        }

        // Pass 3: Sorting Pass
        {
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.sortPipeline);
            pass.setBindGroup(0, this.sortBindGroup0);
            pass.setBindGroup(1, this.sortBindGroup1);
            const numWorkgroups = this.GRID_SIZE.x * this.GRID_SIZE.y;
            pass.dispatchWorkgroups(numWorkgroups);
            pass.end();
        }

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
            pass.setPipeline(this.renderPipeline);
            pass.setBindGroup(0, this.renderBindGroup0);
            pass.setBindGroup(1, this.renderBindGroup1);
            pass.draw(3); // fullscreen triangle
            pass.end();
        }

        this.device.queue.submit([encoder.finish()]);
    }

    updateCameraBuffer(dt) {
        this.camera.update(dt);

        const cameraData = new Float32Array(48);
        cameraData.set(this.camera.vMatrix, 0);
        cameraData.set(this.camera.pMatrix, 16);
        cameraData.set(this.camera.pvMatrix, 32);

        this.device.queue.writeBuffer(this.cameraBuffer, 0, cameraData.buffer);
    }

    updateRenderParams() {
        const renderParams = new Float32Array([this.scaleMultiplier, this.showSfMPoints]);
        this.device.queue.writeBuffer(this.renderParamsBuffer, 0, renderParams.buffer);
    }
}
