const THREADS_PER_WORKGROUP = 256u;

struct TileParams {
    vertexCount : u32,
    gridX : u32,
    gridY : u32,
    maxPerTile : u32,
};

@group(0) @binding(0) var<storage, read> verticesZ : array<f32>;
@group(0) @binding(1) var<storage, read_write> tileIndices : array<u32>;
@group(0) @binding(2) var<storage, read> tileCounters : array<u32>;
@group(0) @binding(3) var<storage, read> params : TileParams;

@compute @workgroup_size(THREADS_PER_WORKGROUP)
fn cs_main(@builtin(local_invocation_id) thread_local_id : vec3<u32>,
           @builtin(workgroup_id) workgroup_id : vec3<u32>) {

    let threadID = thread_local_id.x;
    let tileID = workgroup_id.x;

    let idxCountInTile = min(tileCounters[tileID], params.maxPerTile);
    // empty tile
    if (idxCountInTile == 0u) { return; }

    let baseIdx = tileID * params.maxPerTile;

    // odd-even sort
    // must be done idxCountInTile - 1 iterations to guarantee sorted order
    for (var i = 0u; i < idxCountInTile - 1u; i = i + 1u) {
        let offset = i & 1u; // 0 for even, 1 for odd

        // this amount of pair-wise comparisons must be done this iteration
        let compsPerPass = (idxCountInTile - offset) / 2u;

        // each thread does this many comparisons
        // since we only have THREADS_PER_WORKGROUP threads,
        // some threads have to do multiple comparisons to cover all pairs
        let compsPerThread = (compsPerPass + THREADS_PER_WORKGROUP - 1u) / THREADS_PER_WORKGROUP;

        for (var j = 0u; j < compsPerThread; j = j + 1u) {

            // index inside this tile
            let compIdx = THREADS_PER_WORKGROUP * j + threadID;
            if (compIdx >= compsPerPass) {
                continue;
            }

            // baseIdx gives start of this tile's indices
            // start offsets by 0 or 1 depending on pass
            let leftIdx = baseIdx + offset + compIdx * 2u;
            let rightIdx = leftIdx + 1u;

            let leftVertexIdx = tileIndices[leftIdx];
            let rightVertexIdx = tileIndices[rightIdx];

            let leftZ = verticesZ[leftVertexIdx];
            let rightZ = verticesZ[rightVertexIdx];

            // swap if out of order (farther vertex has larger z in NDC)
            if (leftZ > rightZ) {
                tileIndices[leftIdx] = rightVertexIdx;
                tileIndices[rightIdx] = leftVertexIdx;
            }
        }
        // synchronize all threads before next iteration
        workgroupBarrier();
    }
}