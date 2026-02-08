/* ===== Radix Sort ===== */
/*
    Complexity: O(n), but really fast one.
    It could be better. I ****ed up the implementation.
    Designed for one workgroup per tile.
*/

// threads per workgroup more than 256u
// will not gain more performance
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

@group(1) @binding(3) var<storage, read_write> bucketFlags : array<u32>;
@group(1) @binding(4) var<storage, read_write> dstTileIndices: array<u32>;

// 256 u32 counters for workgroup
var<workgroup> localCounters : array<atomic<u32>, 256u>;

// scan counters for scatter step
var<workgroup> tempCounters : array<u32, 256u>;

@compute @workgroup_size(THREADS_PER_WORKGROUP)
fn cs_main(@builtin(local_invocation_id) thread_local_id : vec3<u32>,
           @builtin(workgroup_id) workgroup_id : vec3<u32>) {
    
    let threadID = thread_local_id.x;
    let tileID = workgroup_id.x;
    let MAX_PER_TILE = uGParams.maxPerTile;

    let countInTile = min(inTileCounters[tileID], uGParams.maxPerTile);
    if (countInTile == 0u) { return; }

    let counterPerThread = (256u + THREADS_PER_WORKGROUP - 1u) / THREADS_PER_WORKGROUP;
    let elementsPerThread = (countInTile + THREADS_PER_WORKGROUP - 1u) / THREADS_PER_WORKGROUP;
    let baseIdx = tileID * MAX_PER_TILE;

    for (var offset = 0u; offset < 32u; offset = offset + 8u) {

        /* ===== Local Counter Reset ===== */
        for (var i = 0u; i < counterPerThread; i = i + 1u) {
            let idx = i * THREADS_PER_WORKGROUP + threadID;
            if (idx >= 256u) { continue; }
            
            atomicStore(&localCounters[idx], 0u);
        }
        workgroupBarrier();

        /* ===== Count ===== */
        for (var i = 0u; i < elementsPerThread; i = i + 1u) {
            let local_idx = i * THREADS_PER_WORKGROUP + threadID;
            
            if (local_idx >= countInTile) { continue; }
            
            let key = depthKeys[inOutTileIndices[baseIdx + local_idx]];
            let bucket = (key >> offset) & 0xFFu;
            bucketFlags[baseIdx + local_idx] = bucket;
            atomicAdd(&localCounters[bucket], 1u);
        }
        workgroupBarrier();

        /* ===== Scan ===== */
        // prefix sum on localCounters
        if (threadID == 0u) {
            var sum = 0u;
            for (var i = 0u; i < 256u; i = i + 1u) {
                let val = atomicLoad(&localCounters[i]);
                atomicStore(&localCounters[i], sum);
                sum = sum + val;
            }
        }
        workgroupBarrier();

        /* ===== Scatter ===== */
        for (var i = 0u; i < counterPerThread; i = i + 1u) {

            let bucket = i * THREADS_PER_WORKGROUP + threadID;

            if (bucket >= 256u) { continue; }

            // each bucket is handled by exactly one thread
            tempCounters[bucket] = 0u;

            let destIdx = atomicLoad(&localCounters[bucket]);
            for (var j = 0u; j < countInTile; j = j + 1u) {

                if (bucketFlags[baseIdx + j] != bucket) { continue; }
                
                let temp = tempCounters[bucket];
                dstTileIndices[baseIdx + destIdx + temp] = inOutTileIndices[baseIdx + j];
                tempCounters[bucket] = temp + 1u;
            }
        }
        workgroupBarrier();

        /* ===== Copy back ===== */
        for (var i = 0u; i < elementsPerThread; i = i + 1u) {
            let local_idx = i * THREADS_PER_WORKGROUP + threadID;

            if (local_idx >= countInTile) { continue; }
            
            inOutTileIndices[baseIdx + local_idx] = dstTileIndices[baseIdx + local_idx];
        }
        workgroupBarrier();
    }

}
