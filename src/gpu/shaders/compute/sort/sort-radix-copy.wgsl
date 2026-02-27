/* !!! DO NOT CHANGE this value !!! */
const THREADS_PER_WORKGROUP = 256u;

struct GlobalParams {
    splatCount : u32,
    gridX : u32,
    gridY : u32,
    maxSortableSplatCount : u32,
};

struct Splat2D {
    pos : vec3<f32>,
    cov : vec3<f32>,
    color : vec4<f32>,
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

@group(1) @binding(0) var<storage, read_write> inKeys : array<Key>;
@group(1) @binding(1) var<storage, read_write> inSplatID : array<u32>;
@group(1) @binding(2) var<storage, read> outKeys : array<Key>;
@group(1) @binding(3) var<storage, read> outSplatID : array<u32>;
@group(1) @binding(4) var<storage, read> sortableSplatCount : SortableSplatCount;

// copy outSplats to inSplats
@compute @workgroup_size(THREADS_PER_WORKGROUP)
fn cs_main(
    @builtin(global_invocation_id) global_id : vec3<u32>
) {
    let index = global_id.x;

    if (index < sortableSplatCount.count && index < uGlobalParams.maxSortableSplatCount) {
        // copy outKeys to inKeys
        inKeys[index] = outKeys[index];
        // copy outSplatID to inSplatID
        inSplatID[index] = outSplatID[index];
    }
}