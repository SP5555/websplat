/* !!! DO NOT CHANGE this value !!! */
const THREADS_PER_WORKGROUP = 256u;

struct SceneParams {
    splatCount : u32,
}

struct Splat2D {
    pos : vec3<f32>,
    cov : vec3<f32>,
    color : vec4<f32>,
};

@group(0) @binding(0) var<uniform> uSceneParams : SceneParams;

@group(1) @binding(0) var<storage, read> inSplats : array<Splat2D>;
@group(1) @binding(1) var<storage, read_write> outSplats : array<Splat2D>;
@group(1) @binding(2) var<storage, read> radixLocalCounters : array<u32>;
@group(1) @binding(3) var<storage, read_write> radixGlobalCounters : array<atomic<u32>>;
@group(1) @binding(4) var<storage, read> radixBucketFlag : array<u32>;

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
    if (splatIndex < uSceneParams.splatCount) {
        let bucket = localBucketFlags[local_id.x];

        var destIndex = atomicLoad(&radixGlobalCounters[bucket]);

        for (var i: u32 = 0u; i < workgroup_id.x; i = i + 1u) {
            destIndex = destIndex + radixLocalCounters[i * 256u + bucket];
        }
    
        for (var i: u32 = 0u; i < local_id.x; i = i + 1u) {
            if (localBucketFlags[i] == bucket) {
                destIndex = destIndex + 1u;
            }
        }
    
        outSplats[destIndex] = inSplats[splatIndex];
    }

}