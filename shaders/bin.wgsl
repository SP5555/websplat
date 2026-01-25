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
@group(0) @binding(3) var<uniform> params : BinParams;

fn computeBinIndex(pos : vec4<f32>) -> u32 {
    // Assuming pos.xy in normalized screen space [-1,1]
    let x = u32(clamp(((pos.x + 1.0) * 0.5) * f32(params.gridX), 0.0, f32(params.gridX - 1)));
    let y = u32(clamp(((pos.y + 1.0) * 0.5) * f32(params.gridY), 0.0, f32(params.gridY - 1)));
    return y * params.gridX + x;
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= params.vertexCount) { return; }

    let v = vertices[i];
    let binIndex = computeBinIndex(v.pos);

    // Atomically increment counter for this bin
    let offset = atomicAdd(&binCounters[binIndex], 1u);

    // Prevent overflow
    if (offset < params.maxPerBin) {
        binVertices[binIndex * params.maxPerBin + offset] = v;
    }
}
