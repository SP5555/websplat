'use strict';

import Camera from "../scene/camera.js";
import { SplatVertexPacker } from "../gaussian/splat-vertex-packer.js";
import WGSLShader from "../gpu/shaders/wgsl-shader.js";

const BUFFER_MIN_SIZE = 80; // bytes

export default class RasterSplatRenderer {
    constructor(input) {
        this.input = input;

        this.rasterShaderPath = './src/gpu/shaders/raster-render/raster-render.wgsl';

        /* ===== Private Zone ===== */
        this.canvas = document.getElementById('canvas00');
        this.device = null;
        this.context = null;
        
        this.camera = new Camera(input, this.canvas.width / this.canvas.height);
        this.vertexCount = 0;
        this.floatsPerVertex = 16;

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

        this.createRenderPipeline();

        this.isPipelineInitialized = true;
    }

    async initShaders() {
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

        this.vertexBuffer = this.device.createBuffer({
            label: "Vertex Buffer",
            size: BUFFER_MIN_SIZE,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
    }

    createRenderPipeline() {
        this.renderPipeline = this.device.createRenderPipeline({
            label: "Raster Splat Render Pipeline",
            layout: 'auto',
            vertex: {
                module: this.rasterShader.getModule(),
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: this.floatsPerVertex * 4,
                    attributes: [{
                        shaderLocation: 0,
                        offset: 0,
                        format: 'float32x3',
                    }],
                }],
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
                }]
            },
            primitive: {
                topology: 'point-list',
                cullMode: 'none',
            },
        });

        this.renderBindGroup = this.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.cameraBuffer } },
            ],
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
        const { vertexCount, floatsPerVertex, bufferData } = SplatVertexPacker.build(meshData);
        this.vertexCount = vertexCount;
        this.floatsPerVertex = floatsPerVertex;

        this.reallocateVertexBuffer(bufferData);

        this.device.queue.writeBuffer(this.vertexBuffer, 0, bufferData.buffer);
    }

    reallocateVertexBuffer(bufferData) {
        if (this.vertexBuffer) this.vertexBuffer.destroy();
        this.vertexBuffer = this.device.createBuffer({
            label: "Vertex Buffer",
            size: bufferData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
    }

    /* ===== ===== Rendering ===== ===== */

    render(dt) {
        if (!this.device || !this.context || !this.isPipelineInitialized) return;
    
        this.updateCameraBuffer(dt);

        const commandEncoder = this.device.createCommandEncoder();

        {
            const pass = commandEncoder.beginRenderPass({
                colorAttachments: [{
                    view: this.context.getCurrentTexture().createView(),
                    clearValue: { r: 0, g: 0, b: 0, a: 1 },
                    loadOp: 'clear',
                    storeOp: 'store'
                }],
            });

            pass.setPipeline(this.renderPipeline);
            pass.setBindGroup(0, this.renderBindGroup);
            pass.setVertexBuffer(0, this.vertexBuffer);
            if (this.vertexCount) {
                pass.draw(this.vertexCount, 3, 0, 0);
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
}