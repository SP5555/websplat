/* ===== Faster variant of Odd-Even Sort ===== */
/*
    Complexity: O(n), simple parallel sorting algorithm.
    Sentinel indices (0xFFFFFFFF) are used for unused slots.
    Designed for one workgroup per tile.
    Uses shared memory to reduce global memory accesses.
    however, can only handle up to MAX_VERTICES_PER_TILE vertices per tile.
*/

const THREADS_PER_WORKGROUP = 128u;
const MAX_VERTICES_PER_TILE = 2048u;

struct GlobalParams {
    vertexCount : u32,
    gridX : u32,
    gridY : u32,
    maxPerTile : u32,
};

@group(0) @binding(0) var<uniform> uGParams : GlobalParams;

@group(1) @binding(0) var<storage, read> inSplatsZ : array<f32>;
@group(1) @binding(1) var<storage, read_write> inOutTileIndices : array<u32>;
@group(1) @binding(2) var<storage, read> inTileCounters : array<u32>;

var<workgroup> localIndices : array<u32, MAX_VERTICES_PER_TILE>;
var<workgroup> localVertZs : array<f32, MAX_VERTICES_PER_TILE>;

@compute @workgroup_size(THREADS_PER_WORKGROUP)
fn cs_main(@builtin(local_invocation_id) thread_local_id : vec3<u32>,
           @builtin(workgroup_id) workgroup_id : vec3<u32>) {

    let threadID = thread_local_id.x;
    let tileID = workgroup_id.x;

    let idxCountInTile = min(inTileCounters[tileID], uGParams.maxPerTile);
    // empty tile
    if (idxCountInTile == 0u) { return; }

    let baseIdx = tileID * uGParams.maxPerTile;

    // load into shared memory from global memory
    for (var i = threadID; i < idxCountInTile; i = i + THREADS_PER_WORKGROUP) {
        let vertexIdx = inOutTileIndices[baseIdx + i];
        localIndices[i] = vertexIdx;
        localVertZs[i] = inSplatsZ[vertexIdx];
    }

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

            // start offsets by 0 or 1 depending on pass
            let leftIdx = offset + compIdx * 2u;
            let rightIdx = leftIdx + 1u;

            let leftVertexIdx = localIndices[leftIdx];
            let rightVertexIdx = localIndices[rightIdx];

            let leftZ = localVertZs[leftIdx];
            let rightZ = localVertZs[rightIdx];

            // swap if out of order (farther vertex has larger z in NDC)
            if (leftZ > rightZ) {
                localIndices[leftIdx] = rightVertexIdx;
                localIndices[rightIdx] = leftVertexIdx;

                localVertZs[leftIdx] = rightZ;
                localVertZs[rightIdx] = leftZ;
            }
        }
        // synchronize all threads before next iteration
        workgroupBarrier();
    }

    // write back to global memory
    for (var i = threadID; i < idxCountInTile; i = i + THREADS_PER_WORKGROUP) {
        inOutTileIndices[baseIdx + i] = localIndices[i];
    }
}