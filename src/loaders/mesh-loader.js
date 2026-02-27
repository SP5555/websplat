"use strict";

import { centerAndScaleGaussianData } from "../gaussian/gaussian-precompute.js";
import JSONLoader from "./json-loader.js";
import PLYLoader from "./ply-loader.js";

export default class MeshLoader {
    constructor() {
        this.loaders = {
            '.ply': new PLYLoader(),
            '.json': new JSONLoader()
        };
    }

    async load(file) {
        const extension = this.getExtension(file.name);
        const loader = this.loaders[extension];

        if (!loader) {
            throw new Error(`Unsupported file extension: ${extension}`);
        }

        const rawData = await loader.load(file);
        return centerAndScaleGaussianData(rawData);
    }

    getExtension(filename) {
        return filename.substring(filename.lastIndexOf('.')).toLowerCase();
    }
}