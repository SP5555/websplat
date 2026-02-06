'use strict';

import GUI from 'lil-GUI';
import { eventBus } from '../utils/event-emitters.js';
import { EVENTS } from '../utils/event.js';

export default class GUIManager {
    constructor(onFileSelected) {
        this.gui = new GUI();

        this.setupGUI();
        this.setupFileInput();
    }

    setupGUI() {
        this.gui.add({ loadShader: () => this.fileInput.click() }, 'loadShader').name('Open PLY File');

        // default values
        const state = {
            fov: 50,
            scaleMultiplier: 1.0,
            showSfM: false,
        };

        this.gui.add(state, 'fov', 10, 80, 1)
            .name('FOV')
            .onChange(value => {
                eventBus.emit(EVENTS.CAMERA_FOV_CHANGE, value);
            });

        this.gui.add(state, 'scaleMultiplier', 0, 1, 0.01)
            .name('Scale Multiplier')
            .onChange(value => {
                eventBus.emit(EVENTS.SCALE_MULTIPLIER_CHANGE, value);
            });

        this.gui.add(state, 'showSfM')
            .name('Show SfM Points')
            .onChange(value => {
                eventBus.emit(EVENTS.SHOW_SFM_CHANGE, value);
            });
    }

    setupFileInput() {
        this.fileInput = document.createElement('input');
        this.fileInput.type = 'file';
        this.fileInput.accept = '.ply';
        this.fileInput.style.display = 'none';
        document.body.appendChild(this.fileInput);

        // when a file is selected
        this.fileInput.addEventListener('change', async () => {
            const file = this.fileInput.files[0];
            if (!file) return;
            eventBus.emit(EVENTS.FILE_LOAD, file);
        });
    }
}
