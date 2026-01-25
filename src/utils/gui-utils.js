'use strict';

export function isMouseOverGUI() {
    return document.querySelector('.gui-block:hover, .lil-gui:hover') !== null;
}