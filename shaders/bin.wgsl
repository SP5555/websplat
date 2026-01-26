struct Vertex {
    pos : vec4<f32>,    // vec3
    cov1 : vec4<f32>,   // vec3
    cov2 : vec4<f32>,   // vec3
    opacity : f32,
    _pad : vec3<f32>,
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

fn computeBinIndex(pos : vec4<f32>) -> i32 {
    // pos.xy in normalized screen space [-1,1]
    let xF = ((pos.x + 1.0) * 0.5) * f32(params.gridX);
    let yF = ((pos.y + 1.0) * 0.5) * f32(params.gridY);

    return i32(yF) * i32(params.gridX) + i32(xF);
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= params.vertexCount) { return; }

    let v = vertices[i];

    // clip space lies -1<Z<1
    if (v.pos.z < -1.0 || v.pos.z > 1.0) {
        return;
    }
    // out of clip space bounds
    if (v.pos.x < -1.0 || v.pos.x >= 1.0 || v.pos.y < -1.0 || v.pos.y >= 1.0) {
        return;
    }

    let binIndex = computeBinIndex(v.pos);

    let offset = atomicAdd(&binCounters[binIndex], 1u);
    if (offset < params.maxPerBin) {
        binVertices[u32(binIndex) * params.maxPerBin + offset] = v;
    }
}
