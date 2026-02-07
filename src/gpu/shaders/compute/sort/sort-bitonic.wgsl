/* ===== Modified version of the Bitonic Sort ===== */
/*
    EACH TILE MUST HAVE A SIZE THAT IS A POWER OF 2.
    Complexity: O((log^2)n), massively parallel sorting algorithm.
    Sentinel indices (0xFFFFFFFF) are used for unused slots.
    Designed for one workgroup per tile.
*/

const THREADS_PER_WORKGROUP = 256u;

struct GlobalParams {
    splatCount : u32,
    gridX : u32,
    gridY : u32,
    maxPerTile : u32,
};

@group(0) @binding(0) var<uniform> uGParams : GlobalParams;

@group(1) @binding(0) var<storage, read> depthKeys : array<u32>;
@group(1) @binding(1) var<storage, read_write> inOutTileIndices : array<u32>;
@group(1) @binding(2) var<storage, read> inTileCounters : array<u32>;

const SENTINEL_IDX = 0xFFFFFFFFu;
const SENTINEL_KEY = 0xFFFFFFFFu;

fn next_pow2(v: u32) -> u32 {
    if (v <= 1u) {
        return 1u;
    }
    return 1u << (32u - countLeadingZeros(v - 1u));
}

fn compare_and_swap(leftIdx: u32, rightIdx: u32) {
    let leftSplatIdx  = inOutTileIndices[leftIdx];
    let rightSplatIdx = inOutTileIndices[rightIdx];

    if (leftSplatIdx == SENTINEL_IDX && rightSplatIdx == SENTINEL_IDX) {
        return; // both sentinels
    }

    let leftZ  = select(SENTINEL_KEY, depthKeys[leftSplatIdx], leftSplatIdx != SENTINEL_IDX);
    let rightZ = select(SENTINEL_KEY, depthKeys[rightSplatIdx], rightSplatIdx != SENTINEL_IDX);

    if (leftZ > rightZ) {
        inOutTileIndices[leftIdx]  = rightSplatIdx;
        inOutTileIndices[rightIdx] = leftSplatIdx;
    }
}

@compute @workgroup_size(THREADS_PER_WORKGROUP)
fn cs_main(@builtin(local_invocation_id) thread_local_id : vec3<u32>,
           @builtin(workgroup_id) workgroup_id : vec3<u32>) {
    
    let threadID = thread_local_id.x;
    let tileID = workgroup_id.x;
    let MAX_PER_TILE = uGParams.maxPerTile;

    let countInTile = min(inTileCounters[tileID], MAX_PER_TILE);
    if (countInTile == 0u) { return; }

    let countPow2 = next_pow2(countInTile);

    // bitonic must operate on the whole array of size countPow2
    // thus, unused index slots are filled with max Uint32 value
    let compsPerThread = (countPow2 + THREADS_PER_WORKGROUP - 1u) / THREADS_PER_WORKGROUP;
    let baseIdx = tileID * MAX_PER_TILE;

    // bitonic sort in global memory
    // local workgroup memory version would be faster
    // but avoided due to size limitations
    var k = 2u;
    while (k <= countPow2) {

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
