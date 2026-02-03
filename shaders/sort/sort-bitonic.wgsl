/* ===== Modified version of the Bitonic Sort ===== */
/*
    Each tile must hold 2^n indices.
    Complexity: O((log^2)n), massively parallel sorting algorithm.
    Sentinel indices (0xFFFFFFFF) are used for unused slots.
    Designed for one workgroup per tile.
*/

const THREADS_PER_WORKGROUP = 256u;

struct VertexZ {
    posZ : f32,
};

struct TileParams {
    vertexCount : u32,
    gridX : u32,
    gridY : u32,
    maxPerTile : u32,
};

@group(0) @binding(0) var<storage, read> verticesZ : array<VertexZ>;
@group(0) @binding(1) var<storage, read_write> tileIndices : array<u32>;
@group(0) @binding(2) var<storage, read> tileCounters : array<u32>;
@group(0) @binding(3) var<storage, read> params : TileParams;

fn compare_and_swap(leftIdx: u32, rightIdx: u32) {
    let leftVertexIdx  = tileIndices[leftIdx];
    let rightVertexIdx = tileIndices[rightIdx];

    let leftZ  = select(1.0, verticesZ[leftVertexIdx].posZ, leftVertexIdx != 0xFFFFFFFF);
    let rightZ = select(1.0, verticesZ[rightVertexIdx].posZ, rightVertexIdx != 0xFFFFFFFF);

    if (leftZ > rightZ) {
        tileIndices[leftIdx]  = rightVertexIdx;
        tileIndices[rightIdx] = leftVertexIdx;
    }
}

@compute @workgroup_size(THREADS_PER_WORKGROUP)
fn cs_main(@builtin(local_invocation_id) thread_local_id : vec3<u32>,
           @builtin(workgroup_id) workgroup_id : vec3<u32>) {
    
    let threadID = thread_local_id.x;
    let tileID = workgroup_id.x;
    let MAX_PER_TILE = params.maxPerTile;

    if (min(tileCounters[tileID], MAX_PER_TILE) == 0u) { return; }

    // bitonic must operate on the whole array of size MAX_PER_TILE
    // thus, unused index slots are filled with max Uint32 value
    let compsPerThread = (MAX_PER_TILE + THREADS_PER_WORKGROUP - 1u) / THREADS_PER_WORKGROUP;
    let baseIdx = tileID * MAX_PER_TILE;

    // bitonic sort in global memory
    var k = 2u;
    while (k <= MAX_PER_TILE) {

        /* ===== FLIP BLOCK ===== */
        // each thread handles multiple comparisons if needed
        for (var i = 0u; i < compsPerThread; i = i + 1u) {

            let idxL = i * THREADS_PER_WORKGROUP + threadID;

            // bitwise XOR magic
            let idxR = idxL ^ (k - 1);

            if (idxL < idxR) {
                compare_and_swap(baseIdx + idxL, baseIdx + idxR);
            }
        }
        workgroupBarrier();

        /* ===== DISPERSE BLOCK ===== */
        var j = k >> 1u;
        while (j > 0u) {

            // each thread handles multiple comparisons if needed
            for (var i = 0u; i < compsPerThread; i = i + 1u) {

                let idxL = i * THREADS_PER_WORKGROUP + threadID;

                // another bitwise XOR magic
                let idxR = idxL ^ j;

                if (idxL < idxR) {
                    compare_and_swap(baseIdx + idxL, baseIdx + idxR);
                }
            }

            workgroupBarrier();
            j = j >> 1u;
        }

        k = k << 1u;
    }
}
