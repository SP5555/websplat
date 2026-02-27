/* !!! DO NOT CHANGE this value !!! */
const THREADS_PER_WORKGROUP = 256u;

struct GlobalParams {
    splatCount : u32,
    gridX : u32,
    gridY : u32,
    maxSortableSplatCount : u32,
};

struct Key {
    tileID : u32,
    depth : u32,
};

struct SortableSplatCount {
    count : u32,
    // padding until 80 bytes
    _padding : array<u32, 19>,
};

@group(0) @binding(0) var<uniform> uGlobalParams : GlobalParams;

@group(1) @binding(0) var<storage, read> inKeys : array<Key>;
@group(1) @binding(1) var<storage, read> inSplatID : array<u32>;
@group(1) @binding(2) var<storage, read_write> outKeys : array<Key>;
@group(1) @binding(3) var<storage, read_write> outSplatID : array<u32>;

@group(1) @binding(4) var<storage, read> radixLocalCounters : array<u32>;
@group(1) @binding(5) var<storage, read_write> radixGlobalCounters : array<u32>;
@group(1) @binding(6) var<storage, read> radixBucketFlag : array<u32>;
@group(1) @binding(7) var<storage, read> sortableSplatCount : SortableSplatCount;

// 256 u32 counters for workgroup
var<workgroup> localCounters : array<atomic<u32>, 256u>;
// bucket flag for each splat
var<workgroup> localBucketFlags : array<u32, THREADS_PER_WORKGROUP>;

// enough workgroups are launched simultaneously to cover all splats
@compute @workgroup_size(THREADS_PER_WORKGROUP)
fn cs_main(
    @builtin(global_invocation_id) global_id : vec3<u32>,
    @builtin(local_invocation_id) local_id : vec3<u32>,
    @builtin(workgroup_id) workgroup_id : vec3<u32>
) {
    // load into local workgroup memory
    localBucketFlags[local_id.x] = radixBucketFlag[global_id.x];
    
    let splatIndex = global_id.x;
    if (splatIndex < sortableSplatCount.count && splatIndex < uGlobalParams.maxSortableSplatCount) {
        let bucket = localBucketFlags[local_id.x];

        var destIndex = radixGlobalCounters[bucket];

        for (var i: u32 = 0u; i < workgroup_id.x; i = i + 1u) {
            destIndex = destIndex + radixLocalCounters[i * 256u + bucket];
        }
    
        for (var i: u32 = 0u; i < local_id.x; i = i + 1u) {
            if (localBucketFlags[i] == bucket) {
                destIndex = destIndex + 1u;
            }
        }
    
        outKeys[destIndex] = inKeys[splatIndex];
        outSplatID[destIndex] = inSplatID[splatIndex];
    }

}