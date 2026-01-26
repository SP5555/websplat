struct Vertex {
    pos : vec3<f32>,
    opacity : f32,
    cov1 : vec3<f32>,
    cov2 : vec3<f32>,
    color : vec3<f32>,
};

struct BinParams {
    vertexCount : u32,
    gridX : u32,
    gridY : u32,
    maxPerBin : u32,
};

@group(0) @binding(0) var<storage, read> vertices : array<Vertex>;
@group(0) @binding(1) var<storage, read_write> binVertices : array<Vertex>;
@group(0) @binding(2) var<storage, read_write> binCounters : array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> params : BinParams;

fn toIndex(x : i32, y : i32) -> u32 {
    return u32(y * i32(params.gridX) + x);
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= params.vertexCount) { return; }

    let v = vertices[i];

    // point outside clip space
    if (abs(v.pos.z) > 1.0 || abs(v.pos.x) > 1.0 || abs(v.pos.y) > 1.0) {
        return;
    }

    // compute bin coordinates as floats
    let xF = ((v.pos.x + 1.0) * 0.5) * f32(params.gridX);
    let yF = ((v.pos.y + 1.0) * 0.5) * f32(params.gridY);

    // integer bin coordinates
    let binX0 = i32(floor(xF));
    let binY0 = i32(floor(yF));

    if (binX0 >= 0 && binX0 < i32(params.gridX) &&
        binY0 >= 0 && binY0 < i32(params.gridY)) {
        let binIndex = toIndex(binX0, binY0);
        let count = atomicAdd(&binCounters[binIndex], 1u);
        if (count < params.maxPerBin) {
            let offset = binIndex * params.maxPerBin + count;
            binVertices[offset] = v;
        }
    }

    // feel free to comment out down here

    // vertical neighbor
    let dy = yF - f32(binY0);
    var verticalY = binY0 + 1;
    if (dy < 0.5) { verticalY = binY0 - 1; };
    if (verticalY >= 0 && verticalY < i32(params.gridY)) {
        let binIndex = toIndex(binX0, verticalY);
        let count = atomicAdd(&binCounters[binIndex], 1u);
        if (count < params.maxPerBin) {
            let offset = binIndex * params.maxPerBin + count;
            binVertices[offset] = v;
        }
    }

    // horizontal neighbor
    let dx = xF - f32(binX0);
    var horizontalX = binX0 + 1;
    if (dx < 0.5) { horizontalX = binX0 - 1; };
    if (horizontalX >= 0 && horizontalX < i32(params.gridX)) {
        let binIndex = toIndex(horizontalX, binY0);
        let count = atomicAdd(&binCounters[binIndex], 1u);
        if (count < params.maxPerBin) {
            let offset = binIndex * params.maxPerBin + count;
            binVertices[offset] = v;
        }
    }

    // diagonal neighbor
    if (verticalY >= 0 && verticalY < i32(params.gridY) &&
        horizontalX >= 0 && horizontalX < i32(params.gridX)) {
        let binIndex = toIndex(horizontalX, verticalY);
        let count = atomicAdd(&binCounters[binIndex], 1u);
        if (count < params.maxPerBin) {
            let offset = binIndex * params.maxPerBin + count;
            binVertices[offset] = v;
        }
    }

}
