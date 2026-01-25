'use strict';

import { mat4, vec3 } from 'gl-matrix';

export default class Camera {
    constructor(input, aspect=1.0) {
        this.input = input;

        this.position = vec3.fromValues(0, 0, 5);
        this.target = vec3.fromValues(0, 0, 0);

        this.forward = null;
        this.up = null;
        this.right = null;

        this.updateViewSpaceVectors();

        this.rotateSpeed = 0.2;
        this.panSpeed = 0.1;
        this.zoomSpeed = 0.1;

        this.minDistance = 0.1;
        this.maxDistance = 50.0;

        this.vMatrix = mat4.create();

        this.fov = 45 * Math.PI / 180; // in radians
        this.aspect = aspect;
        this.near = 0.1;
        this.far = 100.0;
        this.pMatrix = mat4.perspective(mat4.create(), this.fov, this.aspect, this.near, this.far);
    }

    updateViewSpaceVectors() {
        this.forward = vec3.normalize(vec3.create(), vec3.subtract(vec3.create(), this.target, this.position));
        this.right = vec3.normalize(vec3.create(), vec3.cross(vec3.create(), this.forward, vec3.fromValues(0, 1, 0)));
        this.up = vec3.normalize(vec3.create(), vec3.cross(vec3.create(), this.right, this.forward));
    }

    update(dt=0.016) {
        const { dx, dy } = this.input.consumeDelta();
        const scrollDelta = this.input.consumeScroll();

        const distance = vec3.distance(this.position, this.target);

        if (this.input.mouseDownButtons.left && !this.input.shiftPressed) { // rotate
            const offset = vec3.create();
            vec3.subtract(offset, this.position, this.target);

            const x_angle = -dx * this.rotateSpeed * dt;
            const y_angle = -dy * this.rotateSpeed * dt;
            let rotationMatrix = mat4.create();

            rotationMatrix = mat4.rotate(rotationMatrix, rotationMatrix, x_angle, this.up);
            rotationMatrix = mat4.rotate(rotationMatrix, rotationMatrix, y_angle, this.right);

            const rotatedOffset = vec3.transformMat4(vec3.create(), offset, rotationMatrix);
            vec3.add(this.position, this.target, rotatedOffset);
        }

        if (this.input.mouseDownButtons.left && this.input.shiftPressed) { // pan
            let translation = vec3.create();
            vec3.scaleAndAdd(translation, translation, this.right, -dx * distance * this.panSpeed * dt);
            vec3.scaleAndAdd(translation, translation, this.up, dy * distance * this.panSpeed * dt);

            vec3.add(this.position, this.position, translation);
            vec3.add(this.target, this.target, translation);
        }

        if (scrollDelta !== 0) { // zoom
            // Direction from camera to target
            const viewDir = vec3.create();
            vec3.subtract(viewDir, this.target, this.position);
            vec3.normalize(viewDir, viewDir);

            // Scale the direction by scrollDelta and distance
            const distance = vec3.distance(this.position, this.target);
            const translation = vec3.create();
            vec3.scale(translation, viewDir, -scrollDelta * distance * this.zoomSpeed);

            // Apply zoom
            vec3.add(this.position, this.position, translation);

            // Clamp distance
            const newDistance = vec3.distance(this.position, this.target);
            if (newDistance < this.minDistance) {
                vec3.scaleAndAdd(this.position, this.target, viewDir, -this.minDistance);
            }
            if (newDistance > this.maxDistance) {
                vec3.scaleAndAdd(this.position, this.target, viewDir, -this.maxDistance);
            }
        }

        this.updateViewSpaceVectors();
        mat4.lookAt(this.vMatrix, this.position, this.target, this.up);
    }

    updateAspect(aspect) {
        this.aspect = aspect;
        mat4.perspective(this.pMatrix, this.fov, this.aspect, this.near, this.far);
    }
}