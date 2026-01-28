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

@compute @workgroup_size(128)
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

    // integer tile coordinates
    let tileX0 = i32(floor(xF));
    let tileY0 = i32(floor(yF));
    if (tileX0 >= 0 && tileX0 < i32(params.gridX) &&
        tileY0 >= 0 && tileY0 < i32(params.gridY)) {
        tryPush(toIndex(tileX0, tileY0), i);
    }

    // feel free to comment out down here

    // vertical neighbor
    let dy = yF - f32(tileY0);
    var verticalY = tileY0 + 1;
    if (dy < 0.5) { verticalY = tileY0 - 1; };
    if (verticalY >= 0 && verticalY < i32(params.gridY)) {
        tryPush(toIndex(tileX0, verticalY), i);
    }

    // horizontal neighbor
    let dx = xF - f32(tileX0);
    var horizontalX = tileX0 + 1;
    if (dx < 0.5) { horizontalX = tileX0 - 1; };
    if (horizontalX >= 0 && horizontalX < i32(params.gridX)) {
        tryPush(toIndex(horizontalX, tileY0), i);
    }

    // diagonal neighbor
    if (verticalY >= 0 && verticalY < i32(params.gridY) &&
        horizontalX >= 0 && horizontalX < i32(params.gridX)) {
        tryPush(toIndex(horizontalX, verticalY), i);
    }
}
