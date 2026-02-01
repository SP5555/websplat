struct Vertex {
    pos : vec3<f32>,
    opacity : f32,
    cov1 : vec3<f32>,
    cov2 : vec3<f32>,
    color : vec3<f32>,
};

struct TileParams {
    vertexCount : u32,
    gridX : u32,
    gridY : u32,
    maxPerTile : u32,
};

@group(0) @binding(0) var<storage, read> vertices : array<Vertex>;
@group(0) @binding(1) var<storage, read_write> tileIndices : array<u32>;
@group(0) @binding(2) var<storage, read_write> tileCounters : array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> params : TileParams;

fn toIndex(x : i32, y : i32) -> u32 {
    return u32(y * i32(params.gridX) + x);
}

fn tryPush(tileIdx: u32, vertexIdx: u32) {
    loop {
        let old = atomicLoad(&tileCounters[tileIdx]);
        if (old >= params.maxPerTile) {
            return;
        }

        // bruh
        if (atomicCompareExchangeWeak(&tileCounters[tileIdx], old, old + 1u).exchanged) {
            let offset = tileIdx * params.maxPerTile + old;
            tileIndices[offset] = vertexIdx;
            return;
        }
    }
}

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= params.vertexCount) { return; }

    let v = vertices[i];

    // point outside clip space
    if (abs(v.pos.z) > 1.0 || abs(v.pos.x) > 1.0 || abs(v.pos.y) > 1.0) {
        return;
    }

    // compute tile coordinates as floats
    let xF = ((v.pos.x + 1.0) * 0.5) * f32(params.gridX);
    let yF = ((v.pos.y + 1.0) * 0.5) * f32(params.gridY);

    let cxx = v.cov1.x;
    let cxy = v.cov1.y;
    let cyy = v.cov1.z;

    let trace = cxx + cyy;
    let det = cxx * cyy - cxy * cxy;
    if (det <= 0.0) {
        return; // broken, skip
    }
    let temp = trace * trace - 4.0 * det;
    let lambda1 = 0.5 * (trace + sqrt(temp));
    let lambda2 = 0.5 * (trace - sqrt(temp));

    let max = max(lambda1, lambda2);
    let maxRadius = 3.0 * sqrt(max); // 3 sigma

    let minX = v.pos.x - maxRadius;
    let maxX = v.pos.x + maxRadius;
    let minY = v.pos.y - maxRadius;
    let maxY = v.pos.y + maxRadius;

    let x0 = clamp(i32(floor((minX + 1.0)*0.5 * f32(params.gridX))), 0, i32(params.gridX)-1);
    let x1 = clamp(i32(floor((maxX + 1.0)*0.5 * f32(params.gridX))), 0, i32(params.gridX)-1);
    let y0 = clamp(i32(floor((minY + 1.0)*0.5 * f32(params.gridY))), 0, i32(params.gridY)-1);
    let y1 = clamp(i32(floor((maxY + 1.0)*0.5 * f32(params.gridY))), 0, i32(params.gridY)-1);

    for (var ty = y0; ty <= y1; ty = ty + 1) {
        for (var tx = x0; tx <= x1; tx = tx + 1) {
            tryPush(toIndex(tx, ty), i);
        }
    }
}
