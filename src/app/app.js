'use strict';

import Renderer from "../renderer/renderer.js";
import GUIManager from "../gui/gui-manager.js";
import PLYLoader from "../loaders/ply-loader.js";
import { eventBus } from "../utils/event-emitters.js";
import { EVENTS } from "../utils/event.js";
import Input from "../input/input.js";
import { GaussianPrecompute } from "../preprocessing/gaussian-precompute.js";

export default class App {
    constructor() {
        this.input = new Input();
        this.renderer = new Renderer(this.input);
        this.guiManager = new GUIManager();
        this.plyLoader = new PLYLoader();

        this.lastTime = 0;
        
        eventBus.on(EVENTS.FILE_LOAD, async (file) => {
            const meshData = await this.plyLoader.load(file);
            eventBus.emit(EVENTS.MESH_READY, meshData);
        });

        eventBus.on(EVENTS.MESH_READY, (meshData) => {
            this.renderer.setMeshData(GaussianPrecompute(meshData));
        });
    }

    start() {
        requestAnimationFrame(this.loop);
    }

    loop = (currentTime) => {
        const dt = (currentTime - this.lastTime) / 1000;
        this.lastTime = currentTime;

        this.renderer.render(dt);

        requestAnimationFrame(this.loop);
    }
}