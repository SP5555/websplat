struct SortableSplatCount {
    count : u32,
};

@group(1) @binding(0) var<storage, read_write> outTileCounters : array<u32>;
@group(1) @binding(1) var<storage, read_write> outSortableSplatCount : SortableSplatCount;
@group(1) @binding(2) var<storage, read_write> radixGlobalCounter : array<u32>;

@compute @workgroup_size(256)
fn cs_main(
    @builtin(local_invocation_id) lid : vec3<u32>,
    @builtin(workgroup_id) gid : vec3<u32>
) {
    let threadID = lid.x;
    let tileID = gid.x;

    if (tileID == 0u) {
        // group 0 threads reset radix global counter
        radixGlobalCounter[threadID] = 0u;
    }

    if (threadID == 0u) {
        // thread 0s reset their respective tile counters
        outTileCounters[tileID] = 0u;
    }
    
    if (threadID == 0u && tileID == 0u) {
        // group 0 thread 0 resets tile counter
        outSortableSplatCount.count = 0u;
    }
}