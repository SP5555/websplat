'use strict';

import ComputeSplatRenderer from "../renderer/compute-splat-renderer.js";
// import RasterSplatRenderer from "../renderer/raster-splat-renderer.js";
import GUIManager from "../gui/gui-manager.js";
import HUDManager from "../hud/hud-manager.js";
import { eventBus } from "../utils/event-emitters.js";
import { EVENTS } from "../utils/event.js";
import Input from "../input/input.js";
import { computeCovariances } from "../gaussian/gaussian-precompute.js";
import MeshLoader from "../loaders/mesh-loader.js";

export default class App {
    constructor() {
        this.input = new Input();
        this.renderer = new ComputeSplatRenderer(this.input);
        // this.renderer = new RasterSplatRenderer(this.input);
        this.guiManager = new GUIManager();
        this.hud = new HUDManager();
        this.meshLoader = new MeshLoader();

        this.overlayAccumulator = 0;
        this.overlayFrameCount = 0;

        this.lastTime = 0;
        
        eventBus.on(EVENTS.FILE_LOAD, async (file) => {
            const meshData = await this.meshLoader.load(file);
            eventBus.emit(EVENTS.MESH_READY, meshData);
        });

        eventBus.on(EVENTS.MESH_READY, (meshData) => {
            this.renderer.setMeshData(computeCovariances(meshData));
        });
    }

    start() {
        requestAnimationFrame(this.loop);
    }

    loop = (currentTime) => {
        const dt = (currentTime - this.lastTime) / 1000;
        this.lastTime = currentTime;

        this.renderer.render(dt);
        this.updateOverlay(dt);

        requestAnimationFrame(this.loop);
    }

    updateOverlay(dt) {
        this.overlayAccumulator += dt;
        this.overlayFrameCount++;

        if (this.overlayAccumulator >= 0.25) {
            this.hud.updateFPS(this.overlayAccumulator, this.overlayFrameCount);

            this.overlayAccumulator = 0;
            this.overlayFrameCount = 0;
        }
    }
}