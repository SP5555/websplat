/* !!! DO NOT CHANGE this value !!! */
const THREADS_PER_WORKGROUP = 256u;

struct GlobalParams {
    splatCount : u32,
    gridX : u32,
    gridY : u32,
    maxSortableSplatCount : u32,
};

struct RadixParams {
    // 0, 8, 16, 24 for 4 radix sort passes
    bitOffset : u32,
}

struct SortableSplatCount {
    count : u32,
    // padding until 80 bytes
    _padding : array<u32, 19>,
};

struct Key {
    tileID : u32,
    depth : u32,
};

@group(0) @binding(0) var<uniform> uGlobalParams : GlobalParams;
@group(0) @binding(1) var<uniform> uRadixParams : RadixParams;

@group(1) @binding(0) var<storage, read> depthKeys : array<Key>;
@group(1) @binding(1) var<storage, read_write> radixLocalCounters : array<u32>;
@group(1) @binding(2) var<storage, read_write> radixGlobalCounters : array<atomic<u32>>;
@group(1) @binding(3) var<storage, read_write> radixBucketFlag : array<u32>;
@group(1) @binding(4) var<storage, read> sortableSplatCount : SortableSplatCount;

// 256 u32 counters for workgroup
var<workgroup> localCounters : array<atomic<u32>, 256u>;
// bucket flag for each splat
var<workgroup> localBucketFlags : array<u32, THREADS_PER_WORKGROUP>;

fn bucketize(key: Key, bitOffset: u32) -> u32 {
    if (bitOffset < 32u) {
        return (key.depth >> bitOffset) & 0xFFu;
    }
    return (key.tileID >> (bitOffset - 32u)) & 0xFFu;
}

// enough workgroups are launched simultaneously to cover all sortable splats
@compute @workgroup_size(THREADS_PER_WORKGROUP)
fn cs_main(
    @builtin(global_invocation_id) global_id : vec3<u32>,
    @builtin(local_invocation_id) local_id : vec3<u32>
) {

    atomicStore(&localCounters[local_id.x], 0u);
    workgroupBarrier();

    let splatIndex = global_id.x;
    if (splatIndex < sortableSplatCount.count && splatIndex < uGlobalParams.maxSortableSplatCount) {
        // here, each thread works on a single splat
        let bitOffset = uRadixParams.bitOffset;
        // extract relevant bit from depth key
        let key = depthKeys[splatIndex];
        let bucket = bucketize(key, bitOffset);

        atomicAdd(&localCounters[bucket], 1u);
        localBucketFlags[local_id.x] = bucket;
    }

    workgroupBarrier();

    // here, each thread works on a single local counter
    let localCounterVal = atomicLoad(&localCounters[local_id.x]);
    
    // upload and add local counters to global counters
    atomicAdd(&radixGlobalCounters[local_id.x], localCounterVal);

    radixLocalCounters[global_id.x] = localCounterVal;
    radixBucketFlag[global_id.x] = localBucketFlags[local_id.x];
}