/* !!! DO NOT CHANGE this value !!! */
const THREADS_PER_WORKGROUP = 256u;

struct SortableSplatCount {
    count : u32,
}

struct Key {
    tileID : u32,
    depth : u32,
};

@group(1) @binding(0) var<storage, read> depthKeys : array<Key>;
@group(1) @binding(1) var<storage, read> splatIDs : array<u32>;
@group(1) @binding(2) var<storage, read_write> outDepthKeys : array<Key>;
@group(1) @binding(3) var<storage, read_write> outSplatIDs : array<u32>;

@group(1) @binding(4) var<storage, read> radixLocalCounters : array<u32>;
@group(1) @binding(5) var<storage, read> radixGlobalCounters : array<u32>;
@group(1) @binding(6) var<storage, read> radixBucketFlag : array<u32>;
@group(1) @binding(7) var<storage, read> uSortableSplatCount : SortableSplatCount;

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
    workgroupBarrier();
    
    let splatIndex = global_id.x;
    if (splatIndex < uSortableSplatCount.count) {
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

        outDepthKeys[destIndex] = depthKeys[splatIndex];
        outSplatIDs[destIndex] = splatIDs[splatIndex];
    }

}