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
        this.finalRenderPipeline = null;

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
        // await this.createTransformPipeline();
        // await this.createBinPipeline();
        // await this.createSortPipeline();
        await this.createFinalRenderPipeline();
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

    async createTransformPipeline() {
        const shader = new WGSLShader(this.device, './shaders/transform.wgsl');
        await shader.load();

        this.transformInputBuffer = this.device.createBuffer({
            label: "Transform Input Buffer",
            size: 80,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });

        this.transformOutputBuffer = this.device.createBuffer({
            label: "Transform Output Buffer",
            size: 80,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
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

    async createBinPipeline() {
        const shader = new WGSLShader(this.device, './shaders/bin.wgsl');
        await shader.load();

        const GRID_SIZE = { x: 8, y: 8 };
        const MAX_POINTS_PER_BIN = 1024;

        this.binVerticesBuffer = this.device.createBuffer({
            label: "Binned Vertices Buffer",
            size: GRID_SIZE.x * GRID_SIZE.y * MAX_POINTS_PER_BIN * 16 /*floats per vertex*/ * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        this.binCountersBuffer = this.device.createBuffer({
            label: "Bin Counters Buffer",
            size: GRID_SIZE.x * GRID_SIZE.y * 4, // 1x uint32 per bin
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });

        // vertex count in the scene, grid sizes and max vertices a bin can hold
        this.binParamsBuffer = this.device.createBuffer({
            label: "Bin Params Buffer",
            size: 4 * 4, // 4x uint32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        const binParams = new Uint32Array([this.vertexCount, GRID_SIZE.x, GRID_SIZE.y, MAX_POINTS_PER_BIN]);
        this.device.queue.writeBuffer(this.binParamsBuffer, 0, binParams.buffer);

        this.binPipeline = this.device.createComputePipeline({
            label: "Bin Pipeline",
            layout: 'auto',
            compute: {
                module: shader.getModule(),
                entryPoint: 'cs_main'
            }
        });

        this.binBindGroup = this.device.createBindGroup({
            layout: this.binPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.transformOutputBuffer } },
                { binding: 1, resource: { buffer: this.binVerticesBuffer } },
                { binding: 2, resource: { buffer: this.binCountersBuffer } },
                { binding: 3, resource: { buffer: this.binParamsBuffer } }
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
                { binding: 0, resource: { buffer: this.binVerticesBuffer } },
                { binding: 1, resource: { buffer: this.binCountersBuffer } },
                { binding: 2, resource: { buffer: this.binParamsBuffer } }
            ]
        });
    }

    async createFinalRenderPipeline() {
        const shader = new WGSLShader(this.device, './shaders/basic-shader.wgsl');
        await shader.load();

        this.finalRenderPipeline = this.device.createRenderPipeline({
            label: "Render Pipeline",
            layout: 'auto',
            vertex: {
                module: shader.getModule(),
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 16 * 4, // 16 floats per vertex
                    attributes: [
                        { shaderLocation: 0, format: 'float32x4', offset: 0 },     // position
                        { shaderLocation: 1, format: 'float32x4', offset: 4 * 4 }, // covariance part 1
                        { shaderLocation: 2, format: 'float32x4', offset: 8 * 4 }, // covariance part 2
                        { shaderLocation: 3, format: 'float32', offset: 12 * 4 }   // opacity
                    ]
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
            layout: this.finalRenderPipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: this.cameraBuffer } }]
        });

        this.vertexBuffer = this.device.createBuffer({
            label: "Vertex Buffer",
            size: 0,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
    }

    // async createFinalRenderPipeline() {
    //     const shader = new WGSLShader(this.device, './shaders/final-render.wgsl');
    //     await shader.load();

    //     this.finalRenderPipeline = this.device.createRenderPipeline({
    //         label: "Final Render Pipeline",
    //         layout: 'auto',
    //         vertex: {
    //             module: shader.getModule(),
    //             entryPoint: 'vs_main',
    //             buffers: [] // fullscreen triangle
    //         },
    //         fragment: {
    //             module: shader.getModule(),
    //             entryPoint: 'fs_main',
    //             targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }]
    //         },
    //         primitive: { topology: 'triangle-list' }
    //     });

    //     this.finalRenderBindGroup = this.device.createBindGroup({
    //         layout: this.finalRenderPipeline.getBindGroupLayout(0),
    //         entries: [
    //             { binding: 0, resource: { buffer: this.binVerticesBuffer } },
    //             { binding: 1, resource: { buffer: this.binCountersBuffer } },
    //             { binding: 2, resource: { buffer: this.binParamsBuffer } }
    //         ]
    //     });
    // }

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
        const { vertexCount, positions, covariances, opacities } = meshData;
        this.vertexCount = vertexCount;

        // 3 pos + 1 pad + 6 cov + 2 pad + 1 opacity + 3 pad
        const floatsPerVertex = 16;
        const bufferData = new Float32Array(vertexCount * floatsPerVertex);

        // Layout per vertex:
        // [ px, py, pz, ---,
        //   cxx, cxy, cxz, ---,
        //   cyy, cyz, czz, ---,
        //   opacity, ---, ---, --- ]
        for (let i = 0; i < vertexCount; i++) {
            const baseSrc = i * 3;
            const baseCov = i * 6;
            const baseDst = i * floatsPerVertex;

            bufferData[baseDst + 0] = positions[baseSrc + 0];
            bufferData[baseDst + 1] = positions[baseSrc + 1];
            bufferData[baseDst + 2] = positions[baseSrc + 2];

            bufferData[baseDst + 4] = covariances[baseCov + 0];
            bufferData[baseDst + 5] = covariances[baseCov + 1];
            bufferData[baseDst + 6] = covariances[baseCov + 2];

            bufferData[baseDst + 8] = covariances[baseCov + 3];
            bufferData[baseDst + 9] = covariances[baseCov + 4];
            bufferData[baseDst + 10] = covariances[baseCov + 5];

            bufferData[baseDst + 12] = opacities[i];
        }

        this.reallocateVertexBuffer(bufferData.byteLength);
        this.device.queue.writeBuffer(this.vertexBuffer, 0, bufferData);
        // this.device.queue.writeBuffer(this.transformInputBuffer, 0, bufferData);
    }

    reallocateVertexBuffer(size) {
        if (this.vertexBuffer) this.vertexBuffer.destroy();
        this.vertexBuffer = this.device.createBuffer({
            label: "Vertex Buffer",
            size: size,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });

        // if (this.transformInputBuffer) this.transformInputBuffer.destroy();
        // this.transformInputBuffer = this.device.createBuffer({
        //     label: "Transform Input Buffer",
        //     size: size,
        //     usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        // });

        // if (this.transformOutputBuffer) this.transformOutputBuffer.destroy();
        // this.transformOutputBuffer = this.device.createBuffer({
        //     label: "Transform Output Buffer",
        //     size: size,
        //     usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        // });
    }

    /* ===== ===== Rendering ===== ===== */

    render(dt) {
        if (!this.device || !this.context || !this.finalRenderPipeline) return;

        this.updateCameraBuffer(dt);

        const encoder = this.device.createCommandEncoder();
        
        // // Pass 1: Transform Compute Pass
        // {
        //     const pass = encoder.beginComputePass();
        //     pass.setPipeline(this.transformPipeline);
        //     pass.setBindGroup(0, this.transformBindGroup);
        //     const workgroupSize = 64;
        //     const numWorkgroups = Math.min(16, Math.ceil(this.vertexCount / workgroupSize));
        //     pass.dispatchWorkgroups(numWorkgroups);
        //     pass.end();
        // }

        // // Pass 2: Binning Pass
        // {
        //     const pass = encoder.beginComputePass();
        //     pass.setPipeline(this.binPipeline);
        //     pass.setBindGroup(0, this.binBindGroup);
        //     const workgroupSize = 64;
        //     const numWorkgroups = Math.min(16, Math.ceil(this.vertexCount / workgroupSize));
        //     pass.dispatchWorkgroups(numWorkgroups);
        //     pass.end();
        // }

        // // Pass 3: Sorting Pass
        // {
        //     const pass = encoder.beginComputePass();
        //     pass.setPipeline(this.sortPipeline);
        //     pass.setBindGroup(0, this.sortBindGroup);
        //     const GRID_SIZE = { x: 8, y: 8 };
        //     pass.dispatchWorkgroups(GRID_SIZE.x * GRID_SIZE.y);
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
            pass.setBindGroup(0, this.cameraBindGroup);
            pass.setVertexBuffer(0, this.vertexBuffer);
            if (this.vertexCount) pass.draw(this.vertexCount, 1);
            // pass.draw(3); // fullscreen triangle
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
