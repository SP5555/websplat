const TILE_SIZE : u32 = 128u;

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

@group(0) @binding(0) var<storage, read_write> binVertices : array<Vertex>;
@group(0) @binding(1) var<storage, read> binCounters : array<u32>;
@group(0) @binding(2) var<storage, read> params : BinParams;

var<workgroup> sharedVertices : array<Vertex, TILE_SIZE>;

fn min(a: u32, b: u32) -> u32 {
    return select(b, a, a < b);
}

@compute @workgroup_size(64)
fn cs_main(@builtin(workgroup_id) wg_id : vec3<u32>,
           @builtin(local_invocation_id) li_id : vec3<u32>) {

    let binIndex = wg_id.y * params.gridX + wg_id.x;
    let count = binCounters[binIndex];
    if (count == 0u) { return; }

    // total count in bin, capped by maxPerBin for safety
    let total = min(count, params.maxPerBin);
    let baseOffset = binIndex * params.maxPerBin;

    // Tiled processing
    var tileStart = 0u;
    while (tileStart < total) {
        let tileCount = min(TILE_SIZE, total - tileStart);

        // load tile
        for (var i = li_id.x; i < tileCount; i += 64u) {
            sharedVertices[i] = binVertices[baseOffset + tileStart + i];
        }
        workgroupBarrier();

        // sort tile in shared memory (just like before)
        if (li_id.x == 0u) {
            for (var i = 1u; i < tileCount; i++) {
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

        // write tile back
        for (var i = li_id.x; i < tileCount; i += 64u) {
            binVertices[baseOffset + tileStart + i] = sharedVertices[i];
        }

        tileStart += TILE_SIZE;
        workgroupBarrier();
    }
}
