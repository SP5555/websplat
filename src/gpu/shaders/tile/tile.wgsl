struct Splat2D {
    pos : vec3<f32>,
    cov : vec3<f32>,
    color : vec4<f32>,
};

struct GlobalParams {
    splatCount : u32,
    gridX : u32,
    gridY : u32,
    maxPerTile : u32,
};

struct CanvasParams {
    width : u32,
    height : u32,
};

@group(0) @binding(0) var<uniform> uGParams : GlobalParams;
@group(0) @binding(1) var<uniform> uCParams : CanvasParams;

@group(1) @binding(0) var<storage, read> inSplats : array<Splat2D>;
@group(1) @binding(1) var<storage, read_write> outTileIndices : array<u32>;
@group(1) @binding(2) var<storage, read_write> outTileCounters : array<atomic<u32>>;

fn toIndex(x : i32, y : i32) -> u32 {
    return u32(y * i32(uGParams.gridX) + x);
}

fn tryPush(tileIdx: u32, splatIdx: u32) {
    // note: atomicAdd will overshoot the max
    // be sure to clamp the counter afterwards
    let idx = atomicAdd(&outTileCounters[tileIdx], 1u);
    if (idx < uGParams.maxPerTile) {
        outTileIndices[tileIdx * uGParams.maxPerTile + idx] = splatIdx;
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

    let x0 = clamp(i32(floor((minX + 1.0)*0.5 * f32(uGParams.gridX))), 0, i32(uGParams.gridX)-1);
    let x1 = clamp(i32(floor((maxX + 1.0)*0.5 * f32(uGParams.gridX))), 0, i32(uGParams.gridX)-1);
    let y0 = clamp(i32(floor((minY + 1.0)*0.5 * f32(uGParams.gridY))), 0, i32(uGParams.gridY)-1);
    let y1 = clamp(i32(floor((maxY + 1.0)*0.5 * f32(uGParams.gridY))), 0, i32(uGParams.gridY)-1);

    for (var ty = y0; ty <= y1; ty = ty + 1) {
        for (var tx = x0; tx <= x1; tx = tx + 1) {
            tryPush(toIndex(tx, ty), i);
        }
    }
}
