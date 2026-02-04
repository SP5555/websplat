'use strict';

import { mat4, vec3 } from 'gl-matrix';
import { eventBus } from "../utils/event-emitters.js";
import { EVENTS } from "../utils/event.js";
import { isMouseOverGUI } from '../utils/gui-utils.js';

export default class Camera {
    constructor(input, aspect=1.0) {
        this.input = input;

        this.position = vec3.fromValues(0, 0, 3);
        this.target = vec3.fromValues(0, 0, 0);

        this.forward = null;
        this.up = null;
        this.right = null;

        this.updateViewSpaceVectors();

        this.rotateSpeed = 0.1;
        this.panSpeed = 0.1;
        this.zoomSpeed = 0.1;

        this.minDistance = 0.1;
        this.maxDistance = 50.0;

        this.minPitch = -Math.PI / 2 + 0.01;
        this.maxPitch = Math.PI / 2 - 0.01;

        this.vMatrix = mat4.create();

        this.fov = 60 * Math.PI / 180; // in radians
        this.aspect = aspect;
        this.near = 0.01;
        this.far = 50.0;
        this.pMatrix = mat4.perspective(mat4.create(), this.fov, this.aspect, this.near, this.far);

        this.pvMatrix = mat4.create();

        this.initializeEventListeners();

        this.updateMatrices();
    }

    initializeEventListeners() {
        eventBus.on(EVENTS.CAMERA_FOV_CHANGE, (fovDegrees) => {
            this.updateFOV(fovDegrees);
        });
    }

    updateViewSpaceVectors() {
        this.forward = vec3.normalize(vec3.create(), vec3.subtract(vec3.create(), this.target, this.position));
        this.right = vec3.normalize(vec3.create(), vec3.cross(vec3.create(), this.forward, vec3.fromValues(0, 1, 0)));
        this.up = vec3.normalize(vec3.create(), vec3.cross(vec3.create(), this.right, this.forward));
    }

    update(dt=0.016) {
        if (isMouseOverGUI()) return;
        let { dx, dy } = this.input.consumeDelta();
        dx = Math.max(-50, Math.min(50, dx));
        dy = -Math.max(-50, Math.min(50, dy));
        const scrollDelta = this.input.consumeScroll();

        // offset of camera from target
        const offset = vec3.create();
        vec3.subtract(offset, this.position, this.target);
        const viewDir = vec3.normalize(vec3.create(), vec3.negate(vec3.create(), offset));
        const distance = vec3.length(offset);

        let isDirty = false;

        if (this.input.mouseDownButtons.left && !this.input.shiftPressed) { // rotate

            const dir = vec3.normalize(vec3.create(), offset);

            let pitch = Math.asin(dir[1]);
            pitch += dy * this.rotateSpeed * dt;
            pitch = Math.max(this.minPitch, Math.min(this.maxPitch, pitch));

            let yaw = Math.atan2(dir[0], dir[2]);
            yaw += -dx * this.rotateSpeed * dt;
            yaw = yaw % (2 * Math.PI);

            const newDir = vec3.fromValues(
                Math.sin(yaw) * Math.cos(pitch),
                Math.sin(pitch),
                Math.cos(yaw) * Math.cos(pitch)
            );

            const newOffset = vec3.create();
            vec3.scale(newOffset, newDir, distance);
            vec3.add(this.position, this.target, newOffset);

            isDirty = true;
        }

        if (this.input.mouseDownButtons.left && this.input.shiftPressed) { // pan
            let translation = vec3.create();
            vec3.scaleAndAdd(translation, translation, this.right, -dx * distance * this.panSpeed * dt);
            vec3.scaleAndAdd(translation, translation, this.up, dy * distance * this.panSpeed * dt);

            vec3.add(this.position, this.position, translation);
            vec3.add(this.target, this.target, translation);

            isDirty = true;
        }

        if (scrollDelta !== 0) { // zoom
            const distance = vec3.distance(this.position, this.target);
            const translation = vec3.create();
            vec3.scale(translation, viewDir, -scrollDelta * distance * this.zoomSpeed);

            vec3.add(this.position, this.position, translation);

            // Clamp distance
            const newDistance = vec3.distance(this.position, this.target);
            if (newDistance < this.minDistance) {
                vec3.scaleAndAdd(this.position, this.target, viewDir, -this.minDistance);
            }
            if (newDistance > this.maxDistance) {
                vec3.scaleAndAdd(this.position, this.target, viewDir, -this.maxDistance);
            }

            isDirty = true;
        }

        if (!isDirty) return false;
        this.updateViewSpaceVectors();
        this.updateMatrices();
        return true;
    }

    updateMatrices() {
        mat4.lookAt(this.vMatrix, this.position, this.target, this.up);
        mat4.multiply(this.pvMatrix, this.pMatrix, this.vMatrix);
    }

    updateAspect(aspect) {
        this.aspect = aspect;
        mat4.perspective(this.pMatrix, this.fov, this.aspect, this.near, this.far);
        this.updateMatrices();
    }

    updateFOV(fovDegrees) {
        this.fov = fovDegrees * Math.PI / 180;
        mat4.perspective(this.pMatrix, this.fov, this.aspect, this.near, this.far);
        this.updateMatrices();
    }
}