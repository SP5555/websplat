'use strict';

export default class HUDManager {
    constructor() {
        this.fpsElement = document.getElementById('fps-counter');
        this.frameTimeElement = document.getElementById('frame-time-counter');

        this.weightedAvgFPS = 0;
    }

    updateFPS(intervalTime, frameCount) {
        const fps = frameCount / intervalTime;
        this.weightedAvgFPS = this.weightedAvgFPS * 0.6 + fps * 0.4;
        this.fpsElement.textContent = this.weightedAvgFPS.toFixed(1);
        this.frameTimeElement.textContent = (1000 / this.weightedAvgFPS).toFixed(2);
    }
}