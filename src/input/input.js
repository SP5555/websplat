'use strict';

export default class Input {
    constructor() {
        // --- States ---
        this.mouseX = -1;
        this.mouseY = -1;
        this.dx = 0;
        this.dy = 0;
        this.mouseDown = false;
        this.mouseDownButtons = {
            left: false,
            middle: false,
            right: false,
        }
        this.shiftPressed = false;

        this.lastX = 0;
        this.lastY = 0;

        this.scrollDelta = 0;
        this.lastTouchDist = null;

        this.clickCallbacks = [];

        this.isValidClick = false;

        // --- Attach events ---
        this.initMouseEvents();
        this.initTouchEvents();
        this.initKeyboardEvents();
    }

    initMouseEvents() {
        window.addEventListener("mousedown", this.onMouseDown.bind(this));
        window.addEventListener("mouseup", this.onMouseUp.bind(this));
        window.addEventListener("mousemove", this.onMouseMove.bind(this));
        window.addEventListener("wheel", this.onWheel.bind(this));
    }
    initTouchEvents() {
        window.addEventListener("touchstart", this.onTouchStart.bind(this), { passive: false });
        window.addEventListener("touchmove", this.onTouchMove.bind(this), { passive: false });
        window.addEventListener("touchend", this.onTouchEnd.bind(this));
    }

    initKeyboardEvents() {
        window.addEventListener("keydown", (e) => {
            if (e.key === "Shift") this.shiftPressed = true;
        });
        window.addEventListener("keyup", (e) => {
            if (e.key === "Shift") this.shiftPressed = false;
        });
    }
    
    // --------------------
    // Mouse event handlers
    // --------------------
    onMouseDown(e) {
        this.mouseDown = true;

        if (e.button === 0) this.mouseDownButtons.left = true;
        if (e.button === 1) this.mouseDownButtons.middle = true;
        if (e.button === 2) this.mouseDownButtons.right = true;

        this.isValidClick = true;
    }

    onMouseUp(e) {
        this.mouseDown = false;

        if (e.button === 0) this.mouseDownButtons.left = false;
        if (e.button === 1) this.mouseDownButtons.middle = false;
        if (e.button === 2) this.mouseDownButtons.right = false;

        if (this.isValidClick) {
            this.clickCallbacks.forEach(cb => cb(e));
        }

        this.dx = 0;
        this.dy = 0;
    }

    onMouseMove(e) {
        if (this.mouseDown) {
            this.dx = e.clientX - this.lastX;
            this.dy = e.clientY - this.lastY;
            const distSq = this.dx * this.dx + this.dy * this.dy;
            if (distSq > 0) this.isValidClick = false;
        } else {
            this.dx = 0;
            this.dy = 0;
        }

        this.lastX = e.clientX;
        this.lastY = e.clientY;

        const halfW = window.innerWidth / 2;
        const halfH = window.innerHeight / 2;
        this.mouseX = (e.clientX - halfW) / halfW;
        this.mouseY = (e.clientY - halfH) / halfH;
    }

    onWheel(e) {
        this.scrollDelta += Math.sign(e.deltaY);
    }

    onClick(cb) {
        this.clickCallbacks.push(cb);
    }

    // --------------------
    // Touch event handlers
    // --------------------
    onTouchStart(e) {
        this.mouseDown = true;
        if (e.touches.length === 1) {
            const t = e.touches[0];
            this.lastX = t.clientX;
            this.lastY = t.clientY;
        }
        if (e.touches.length === 2) {
            this.lastTouchDist = this.getTouchDistance(e.touches);
        }
    }

    onTouchEnd(e) {
        this.mouseDown = false;
        this.dx = 0;
        this.dy = 0;
        this.lastTouchDist = null;
    }

    onTouchMove(e) {
        e.preventDefault();

        const t = e.touches[0];

        this.dx = t.clientX - this.lastX;
        this.dy = t.clientY - this.lastY;
        this.lastX = t.clientX;
        this.lastY = t.clientY;

        const halfW = window.innerWidth / 2;
        const halfH = window.innerHeight / 2;
        this.mouseX = (t.clientX - halfW) / halfW;
        this.mouseY = (t.clientY - halfH) / halfH;

        if (e.touches.length === 2) {
            const newDist = this.getTouchDistance(e.touches);
            if (this.lastTouchDist !== null) {
                this.scrollDelta -= Math.sign(newDist - this.lastTouchDist);
            }
            this.lastTouchDist = newDist;
        }
    }

    // --------------------
    // Utilities
    // --------------------
    getTouchDistance(touches) {
        const dx = touches[0].clientX - touches[1].clientX;
        const dy = touches[0].clientY - touches[1].clientY;
        return Math.sqrt(dx * dx + dy * dy);
    }

    consumeDelta() {
        const { dx, dy } = this;
        this.dx = 0;
        this.dy = 0;
        return { dx, dy };
    }

    consumeScroll() {
        const delta = this.scrollDelta;
        this.scrollDelta = 0;
        return delta;
    }

    getMousePos() {
        return {
            mouseX: this.mouseX,
            mouseY: this.mouseY,
        }
    }
}