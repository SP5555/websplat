struct Vertex {
    pos : vec4<f32>,
    cov1 : vec4<f32>,
    cov2 : vec4<f32>,
    opacity : f32,
    _pad : vec3<f32>,
};

struct BinParams {
    vertexCount : u32,
    gridX : u32,
    gridY : u32,
    maxPerBin : u32,
};

@group(0) @binding(0) var<storage, read_write> binVertices : array<Vertex>;
@group(0) @binding(1) var<storage, read> binCounters : array<u32>;
@group(0) @binding(2) var<uniform> params : BinParams;

// shared memory for one bin
// WARNING: size must fit in workgroup memory, adjust if needed
var<workgroup> sharedVertices : array<Vertex, 1024>;

fn min(a: u32, b: u32) -> u32 {
    return select(b, a, a < b);
}

@compute @workgroup_size(64)
fn cs_main(@builtin(workgroup_id) wg_id : vec3<u32>,
           @builtin(local_invocation_id) li_id : vec3<u32>) {

    let binIndex = wg_id.x;
    let count = binCounters[binIndex];
    if (count == 0u) { return; }

    // limit to shared memory size
    let localCount = min(count, 1024u);

    // load vertices into shared memory
    for (var i = li_id.x; i < localCount; i += 64u) {
        sharedVertices[i] = binVertices[binIndex * params.maxPerBin + i];
    }
    workgroupBarrier();

    // insertion sort by pos.z (thread 0 does the sorting)
    if (li_id.x == 0u) {
        for (var i = 1u; i < localCount; i++) {
            var key = sharedVertices[i];
            var j = i;
            while (j > 0u && sharedVertices[j-1u].pos.z > key.pos.z) {
                sharedVertices[j] = sharedVertices[j-1u];
                j = j - 1u;
            }
            sharedVertices[j] = key;
        }
    }
    workgroupBarrier();

    // write back sorted vertices
    for (var i = li_id.x; i < localCount; i += 64u) {
        binVertices[binIndex * params.maxPerBin + i] = sharedVertices[i];
    }
}
