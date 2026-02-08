/* !!! DO NOT CHANGE this value !!! */
const THREADS_PER_WORKGROUP = 256u;

struct SortableSplatCount {
    count : u32,
}

struct Key {
    tileID : u32,
    depth : u32,
};

@group(1) @binding(0) var<storage, read> srcDepthKeys : array<Key>;
@group(1) @binding(1) var<storage, read> srcSplatIDs : array<u32>;
@group(1) @binding(2) var<storage, read_write> dstDepthKeys : array<Key>;
@group(1) @binding(3) var<storage, read_write> dstSplatIDs : array<u32>;
@group(1) @binding(4) var<storage, read> uSortableSplatCount : SortableSplatCount;

// copy outSplats to inSplats
@compute @workgroup_size(THREADS_PER_WORKGROUP)
fn cs_main(
    @builtin(global_invocation_id) global_id : vec3<u32>
) {
    let index = global_id.x;

    if (index < uSortableSplatCount.count) {
        dstDepthKeys[index] = srcDepthKeys[index];
        dstSplatIDs[index] = srcSplatIDs[index];
    }
}