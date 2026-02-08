struct Splat2D {
    pos : vec3<f32>,
    cov : vec3<f32>,
    color : vec4<f32>,
};

struct GlobalParams {
    splatCount : u32,
    gridX : u32,
    gridY : u32,
    maxSortableSplatCount : u32,
};

struct CanvasParams {
    width : u32,
    height : u32,
};

struct SortableSplatCount {
    count : atomic<u32>,
};

struct Key {
    tileID : u32,
    depth : u32,
};

@group(0) @binding(0) var<uniform> uGParams : GlobalParams;
@group(0) @binding(1) var<uniform> uCParams : CanvasParams;

@group(1) @binding(0) var<storage, read> inSplats : array<Splat2D>;
@group(1) @binding(1) var<storage, read_write> outTileCounters : array<atomic<u32>>;
@group(1) @binding(2) var<storage, read_write> depthKeys : array<Key>;
@group(1) @binding(3) var<storage, read_write> splatIDs : array<u32>;
@group(1) @binding(4) var<storage, read_write> sortableSplatCount : SortableSplatCount;

fn f32_to_sortable_u32(x: f32) -> u32 {
    let b = bitcast<u32>(x);

    // positive floats: flip sign bit
    // negative floats: invert all bits
    return select(~b, b ^ 0x80000000u, (b >> 31u) == 0u);
}

fn make_key(tileID: u32, depth: f32) -> Key {
    let b = bitcast<u32>(depth);

    return Key(tileID, f32_to_sortable_u32(depth));
}

fn toIndex(x : u32, y : u32) -> u32 {
    return u32(y * uGParams.gridX + x);
}

fn tryPush(tileIdx: u32, splatIdx: u32) {
    let idx = atomicAdd(&sortableSplatCount.count, 1u);

    if (idx < uGParams.maxSortableSplatCount) {
        let key = make_key(tileIdx, inSplats[splatIdx].pos.z);
        atomicAdd(&outTileCounters[tileIdx], 1u);
        depthKeys[idx] = key;
        splatIDs[idx] = splatIdx;
    }
}

@compute @workgroup_size(128)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= uGParams.splatCount) { return; }

    let s = inSplats[i];

    // point outside clip space
    if (abs(s.pos.z) > 1.0 || abs(s.pos.x) > 1.0 || abs(s.pos.y) > 1.0) {
        return;
    }

    // compute 3 sigma "radius" of the ellipse 
    let cxx = s.cov.x;
    let cxy = s.cov.y;
    let cyy = s.cov.z;

    let trace = cxx + cyy;
    let det = cxx * cyy - cxy * cxy;
    if (det <= 0.0) {
        return; // broken, skip
    }
    let temp = trace * trace - 4.0 * det;
    let lambda1 = 0.5 * (trace + sqrt(temp));
    let lambda2 = 0.5 * (trace - sqrt(temp));

    // 3 sigma min and max radius
    let minRadius = 3.0 * sqrt(min(lambda1, lambda2));
    let maxRadius = 3.0 * sqrt(max(lambda1, lambda2));

    // pixel size in NDC space
    let pixelSizeX = 2.0 / f32(uCParams.width);
    let pixelSizeY = 2.0 / f32(uCParams.height);
    let pixelSize = max(pixelSizeX, pixelSizeY);

    // this is too aggressive, commented out for now
    // discard if the splat is too "thin"
    // if (2.0 * minRadius < pixelSize) {
    //     return;
    // }

    // discard if the entire splat is smaller than a pixel
    if (maxRadius < pixelSize) {
        return;
    }

    let minX = s.pos.x - maxRadius;
    let maxX = s.pos.x + maxRadius;
    let minY = s.pos.y - maxRadius;
    let maxY = s.pos.y + maxRadius;

    let x0 = clamp(u32(floor((minX + 1.0)*0.5 * f32(uGParams.gridX))), 0u, uGParams.gridX-1u);
    let x1 = clamp(u32(floor((maxX + 1.0)*0.5 * f32(uGParams.gridX))), 0u, uGParams.gridX-1u);
    let y0 = clamp(u32(floor((minY + 1.0)*0.5 * f32(uGParams.gridY))), 0u, uGParams.gridY-1u);
    let y1 = clamp(u32(floor((maxY + 1.0)*0.5 * f32(uGParams.gridY))), 0u, uGParams.gridY-1u);

    for (var ty = y0; ty <= y1; ty = ty + 1) {
        for (var tx = x0; tx <= x1; tx = tx + 1) {
            tryPush(toIndex(tx, ty), i);
        }
    }
}
