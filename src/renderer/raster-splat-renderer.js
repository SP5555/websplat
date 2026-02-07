'use strict';

import Camera from "../scene/camera.js";
import { SplatDataPacker } from "../gaussian/splat-data-packer.js";
import WGSLShader from "../gpu/shaders/wgsl-shader.js";
import { EVENTS } from "../utils/event.js";
import { eventBus } from "../utils/event-emitters.js";

const BUFFER_MIN_SIZE = 80; // bytes

/* Can't handle Z-depth correctly yet */
/* maybe even impossible without hacks? */

export default class RasterSplatRenderer {
    constructor(input) {
        this.input = input;

        this.transformShaderPath = './src/gpu/shaders/raster/transform/transform.wgsl';
        this.sortShader0Path     = './src/gpu/shaders/raster/sort/sort-radix-count.wgsl';
        this.sortShader1Path     = './src/gpu/shaders/raster/sort/sort-radix-scan.wgsl';
        this.sortShader2Path     = './src/gpu/shaders/raster/sort/sort-radix-scatter.wgsl';
        this.sortShader3Path     = './src/gpu/shaders/raster/sort/sort-radix-copy.wgsl';
        this.rasterShaderPath    = './src/gpu/shaders/raster/render/render.wgsl';

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
        this.transformPipeline = null;
        this.sortPipeline0 = null;
        this.sortPipeline1 = null;
        this.sortPipeline2 = null;
        this.sortPipeline3 = null;
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

        this.createTransformPipeline();
        this.createSortPipeline();
        this.createRenderPipeline();

        this.isPipelineInitialized = true;
    }

    async initShaders() {
        this.transformShader = new WGSLShader(this.device, this.transformShaderPath);
        await this.transformShader.load();

        this.sortShader0 = new WGSLShader(this.device, this.sortShader0Path);
        await this.sortShader0.load();

        this.sortShader1 = new WGSLShader(this.device, this.sortShader1Path);
        await this.sortShader1.load();

        this.sortShader2 = new WGSLShader(this.device, this.sortShader2Path);
        await this.sortShader2.load();

        this.sortShader3 = new WGSLShader(this.device, this.sortShader3Path);
        await this.sortShader3.load();

        this.rasterShader = new WGSLShader(this.device, this.rasterShaderPath);
        await this.rasterShader.load();
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

        // buffer that holds the input splat 3D data for the transform pass
        this.splat3DBuffer = this.device.createBuffer({
            label: "Splat 3D Buffer",
            size: BUFFER_MIN_SIZE,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });

        // buffer that holds the transformed splat 2D data from the transform pass
        this.splat2DBuffer = this.device.createBuffer({
            label: "Splat 2D Buffer",
            size: BUFFER_MIN_SIZE,
            usage: GPUBufferUsage.STORAGE
        });

        // buffer that holds key values for depth sorting
        this.depthKeyBuffer = this.device.createBuffer({
            label: "Transform Output PosZ Buffer",
            size: BUFFER_MIN_SIZE,
            usage: GPUBufferUsage.STORAGE
        });

        /* ===== Radix Sort Buffers ===== */
        this.sortOutSplat2DBuffer = this.device.createBuffer({
            label: "Sort Out Splat 2D Buffer",
            size: BUFFER_MIN_SIZE,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });

        this.radixParamsBuffer = this.device.createBuffer({
            label: "Radix Params Buffer",
            size: 4, // 1x u32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // buffer for radix sort counters
        // each workgroup has 256 u32 local counters
        this.radixLocalCounterBuffer = this.device.createBuffer({
            label: "Radix Counter Buffer",
            size: BUFFER_MIN_SIZE,
            usage: GPUBufferUsage.STORAGE
        });

        // 256 counters of u32 for 8 bits field
        this.radixGlobalCounterBuffer = this.device.createBuffer({
            label: "Radix Global Counter Buffer",
            size: 256 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        // this tracks the buckets the elements belong to
        // used in scan pass to determine local bucket offsets
        this.radixBucketFlagBuffer = this.device.createBuffer({
            label: "Radix Bucket Flag Buffer",
            size: BUFFER_MIN_SIZE,
            usage: GPUBufferUsage.STORAGE
        });
        /* ===== ===== ===== ===== */

        this.sceneParamsBuffer = this.device.createBuffer({
            label: "Scene Params Buffer",
            size: 4, // 1x u32 for splat count
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        const sceneParams = new Uint32Array([this.splatCount]);
        this.device.queue.writeBuffer(this.sceneParamsBuffer, 0, sceneParams.buffer);

        // canvas width and height
        // COPY_DST for canvas resize updates
        this.canvasParamsBuffer = this.device.createBuffer({
            label: "Canvas Params Buffer",
            size: 2 * 4, // width and height as uint32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        const canvasParams = new Uint32Array([this.canvas.width, this.canvas.height]);
        this.device.queue.writeBuffer(this.canvasParamsBuffer, 0, canvasParams.buffer);

        this.depthTexture = this.device.createTexture({
            size: [this.canvas.width, this.canvas.height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
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
                { binding: 0, resource: { buffer: this.splat3DBuffer } },
                { binding: 1, resource: { buffer: this.splat2DBuffer } },
                { binding: 2, resource: { buffer: this.depthKeyBuffer } }
            ]
        });
    }

    createSortPipeline() {
        this.sortPipelineCount = this.device.createComputePipeline({
            label: "Sort Pipeline: Radix Count",
            layout: 'auto',
            compute: {
                module: this.sortShader0.getModule(),
                entryPoint: 'cs_main'
            }
        });

        this.sortPipelineScan = this.device.createComputePipeline({
            label: "Sort Pipeline: Radix Global Counter Scan",
            layout: 'auto',
            compute: {
                module: this.sortShader1.getModule(),
                entryPoint: 'cs_main'
            }
        });

        this.sortPipelineScatter = this.device.createComputePipeline({
            label: "Sort Pipeline: Radix Scatter",
            layout: 'auto',
            compute: {
                module: this.sortShader2.getModule(),
                entryPoint: 'cs_main'
            }
        });

        this.sortPipelineCopy = this.device.createComputePipeline({
            label: "Sort Pipeline: Radix Copy",
            layout: 'auto',
            compute: {
                module: this.sortShader3.getModule(),
                entryPoint: 'cs_main'
            }
        });

        this.sortBindGroupCount0 = this.device.createBindGroup({
            layout: this.sortPipelineCount.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.sceneParamsBuffer } },
                { binding: 1, resource: { buffer: this.radixParamsBuffer } }
            ]
        });

        this.sortBindGroupCount1 = this.device.createBindGroup({
            layout: this.sortPipelineCount.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.depthKeyBuffer } },
                { binding: 1, resource: { buffer: this.radixLocalCounterBuffer } },
                { binding: 2, resource: { buffer: this.radixGlobalCounterBuffer } },
                { binding: 3, resource: { buffer: this.radixBucketFlagBuffer } }
            ]
        });

        this.sortBindGroupScan1 = this.device.createBindGroup({
            layout: this.sortPipelineScan.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.radixGlobalCounterBuffer } },
            ]
        });

        this.sortBindGroupScatter0 = this.device.createBindGroup({
            layout: this.sortPipelineScatter.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.sceneParamsBuffer } }
            ]
        });

        this.sortBindGroupScatter1 = this.device.createBindGroup({
            layout: this.sortPipelineScatter.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.splat2DBuffer } },
                { binding: 1, resource: { buffer: this.sortOutSplat2DBuffer } },
                { binding: 2, resource: { buffer: this.radixLocalCounterBuffer } },
                { binding: 3, resource: { buffer: this.radixGlobalCounterBuffer } },
                { binding: 4, resource: { buffer: this.radixBucketFlagBuffer } }
            ]
        });

        this.sortBindGroupCopy0 = this.device.createBindGroup({
            layout: this.sortPipelineCopy.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.sceneParamsBuffer } },
            ]
        });

        this.sortBindGroupCopy1 = this.device.createBindGroup({
            layout: this.sortPipelineCopy.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.splat2DBuffer } },
                { binding: 1, resource: { buffer: this.sortOutSplat2DBuffer } },
                { binding: 2, resource: { buffer: this.depthKeyBuffer } }
            ]
        });
    }

    createRenderPipeline() {
        this.renderPipeline = this.device.createRenderPipeline({
            label: "Raster Splat Render Pipeline",
            layout: 'auto',
            vertex: {
                module: this.rasterShader.getModule(),
                entryPoint: 'vs_main',
                buffers: [],
            },
            fragment: {
                module: this.rasterShader.getModule(),
                entryPoint: 'fs_main',
                targets: [{
                    format: navigator.gpu.getPreferredCanvasFormat(),
                    blend: {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one-minus-src-alpha',
                            operation: 'add',
                        },
                        alpha: {
                            srcFactor: 'one',
                            dstFactor: 'one-minus-src-alpha',
                            operation: 'add',
                        },
                    },
                    writeMask: GPUColorWrite.ALL,
                }]
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'none',
            },
            depthStencil: {
                format: 'depth24plus',
                depthWriteEnabled: false,
                depthCompare: 'less',
            },
        });

        this.renderBindGroup0 = this.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.canvasParamsBuffer } },
            ],
        });

        this.renderBindGroup1 = this.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.splat2DBuffer } },
            ],
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

        const canvasParams = new Uint32Array([this.canvas.width, this.canvas.height]);
        this.device.queue.writeBuffer(this.canvasParamsBuffer, 0, canvasParams.buffer);
    }

    resizeCanvas() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;

        if (this.depthTexture) {
            this.depthTexture.destroy();
            this.depthTexture = this.device.createTexture({
                size: [this.canvas.width, this.canvas.height],
                format: 'depth24plus',
                usage: GPUTextureUsage.RENDER_ATTACHMENT,
            });
        }
    }
    
    /* ===== ===== Mesh Management ===== ===== */

    setMeshData(meshData) {
        const { splatCount, bufferData } = SplatDataPacker.build(meshData);
        this.splatCount = splatCount;

        this.reallocateBuffers(splatCount);

        this.device.queue.writeBuffer(this.splat3DBuffer, 0, bufferData.buffer);

        const sceneParams = new Uint32Array([this.splatCount]);
        this.device.queue.writeBuffer(this.sceneParamsBuffer, 0, sceneParams.buffer);
    }

    reallocateBuffers(splatCount) {
        if (this.splat3DBuffer) this.splat3DBuffer.destroy();
        this.splat3DBuffer = this.device.createBuffer({
            label: "Splat 3D Buffer",
            size: Math.max(BUFFER_MIN_SIZE, splatCount * this.FLOATS_PER_SPLAT3D * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });

        if (this.splat2DBuffer) this.splat2DBuffer.destroy();
        this.splat2DBuffer = this.device.createBuffer({
            label: "Splat 2D Buffer",
            size: Math.max(BUFFER_MIN_SIZE, splatCount * this.FLOATS_PER_SPLAT2D * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        if (this.sortOutSplat2DBuffer) this.sortOutSplat2DBuffer.destroy();
        this.sortOutSplat2DBuffer = this.device.createBuffer({
            label: "Sort Out Splat 2D Buffer",
            size: Math.max(BUFFER_MIN_SIZE, splatCount * this.FLOATS_PER_SPLAT2D * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });

        const RADIX_THREADS_PER_WORKGROUP = 256; // must match sort-radix.wgsl
        if (this.radixLocalCounterBuffer) this.radixLocalCounterBuffer.destroy();
        if (this.radixBucketFlagBuffer) this.radixBucketFlagBuffer.destroy();
        this.radixLocalCounterBuffer = this.device.createBuffer({
            label: "Radix Local Counter Buffer",
            // each workgroup has 256 u32 local counters
            size: Math.ceil(splatCount / RADIX_THREADS_PER_WORKGROUP) * 256 * 4,
            usage: GPUBufferUsage.STORAGE
        });
        this.radixBucketFlagBuffer = this.device.createBuffer({
            label: "Radix Bucket Flag Buffer",
            // each splat belongs to one radix bucket indicated by a u32 value
            size: splatCount * 4,
            usage: GPUBufferUsage.STORAGE
        });

        this.transformBindGroup1 = this.device.createBindGroup({
            layout: this.transformPipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.splat3DBuffer } },
                { binding: 1, resource: { buffer: this.splat2DBuffer } },
                { binding: 2, resource: { buffer: this.depthKeyBuffer } }
            ]
        });

        this.sortBindGroupCount1 = this.device.createBindGroup({
            layout: this.sortPipelineCount.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.depthKeyBuffer } },
                { binding: 1, resource: { buffer: this.radixLocalCounterBuffer } },
                { binding: 2, resource: { buffer: this.radixGlobalCounterBuffer } },
                { binding: 3, resource: { buffer: this.radixBucketFlagBuffer } }
            ]
        });

        this.sortBindGroupScatter1 = this.device.createBindGroup({
            layout: this.sortPipelineScatter.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.splat2DBuffer } },
                { binding: 1, resource: { buffer: this.sortOutSplat2DBuffer } },
                { binding: 2, resource: { buffer: this.radixLocalCounterBuffer } },
                { binding: 3, resource: { buffer: this.radixGlobalCounterBuffer } },
                { binding: 4, resource: { buffer: this.radixBucketFlagBuffer } }
            ]
        });

        this.sortBindGroupCopy1 = this.device.createBindGroup({
            layout: this.sortPipelineCopy.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.splat2DBuffer } },
                { binding: 1, resource: { buffer: this.sortOutSplat2DBuffer } },
                { binding: 2, resource: { buffer: this.depthKeyBuffer } }
            ]
        });

        this.renderBindGroup1 = this.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this.splat2DBuffer } },
            ],
        });
    }

    /* ===== ===== Rendering ===== ===== */

    render(dt) {
        if (!this.device || !this.context || !this.isPipelineInitialized) return;
    
        this.updateCameraBuffer(dt);

        const commandEncoder = this.device.createCommandEncoder();

        // Pass 1: Transform Compute Pass
        {
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.transformPipeline);
            pass.setBindGroup(0, this.transformBindGroup0);
            pass.setBindGroup(1, this.transformBindGroup1);
            const workgroupSize = 128;
            const numWorkgroups = Math.max(8, Math.ceil(this.splatCount / workgroupSize));
            pass.dispatchWorkgroups(numWorkgroups);
            pass.end();
        }

        // Pass 2: Radix Sort Compute Pass
        {
            for (let offset = 16; offset >= 0; offset -= 8) {
                const RadixParams = new Uint32Array([offset]);
                this.device.queue.writeBuffer(this.radixParamsBuffer, 0, RadixParams);
                {
                    const pass = commandEncoder.beginComputePass();
                    pass.setPipeline(this.sortPipelineCount);
                    pass.setBindGroup(0, this.sortBindGroupCount0);
                    pass.setBindGroup(1, this.sortBindGroupCount1);
                    const workgroupSize = 256;
                    const numWorkgroups = Math.max(8, Math.ceil(this.splatCount / workgroupSize));
                    pass.dispatchWorkgroups(numWorkgroups);
                    pass.end();
                }
                {
                    const pass = commandEncoder.beginComputePass();
                    pass.setPipeline(this.sortPipelineScan);
                    pass.setBindGroup(1, this.sortBindGroupScan1);
                    pass.dispatchWorkgroups(1);
                    pass.end();
                }
                {
                    const pass = commandEncoder.beginComputePass();
                    pass.setPipeline(this.sortPipelineScatter);
                    pass.setBindGroup(0, this.sortBindGroupScatter0);
                    pass.setBindGroup(1, this.sortBindGroupScatter1);
                    const workgroupSize = 256;
                    const numWorkgroups = Math.max(8, Math.ceil(this.splatCount / workgroupSize));
                    pass.dispatchWorkgroups(numWorkgroups);
                    pass.end();
                }
                {
                    const pass = commandEncoder.beginComputePass();
                    pass.setPipeline(this.sortPipelineCopy);
                    pass.setBindGroup(0, this.sortBindGroupCopy0);
                    pass.setBindGroup(1, this.sortBindGroupCopy1);
                    const workgroupSize = 256;
                    const numWorkgroups = Math.max(8, Math.ceil(this.splatCount / workgroupSize));
                    pass.dispatchWorkgroups(numWorkgroups);
                    pass.end();
                }
            }
        }

        // Pass 3: Raster Render Pass
        {
            const pass = commandEncoder.beginRenderPass({
                colorAttachments: [{
                    view: this.context.getCurrentTexture().createView(),
                    clearValue: { r: 0, g: 0, b: 0, a: 1 },
                    loadOp: 'clear',
                    storeOp: 'store'
                }],
                depthStencilAttachment: {
                    view: this.depthTexture.createView(),
                    depthLoadOp: 'clear',
                    depthClearValue: 1.0,
                    depthStoreOp: 'store',
                },
            });

            pass.setPipeline(this.renderPipeline);
            pass.setBindGroup(0, this.renderBindGroup0);
            pass.setBindGroup(1, this.renderBindGroup1);
            if (this.splatCount) {
                // draw call: 6 vertices per splat quad
                pass.draw(6, this.splatCount, 0, 0);
            }
            pass.end();
        }

        this.device.queue.submit([commandEncoder.finish()]);
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