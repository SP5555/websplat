const THREADS_PER_WORKGROUP = 256u;

struct GlobalParams {
    vertexCount : u32,
    gridX : u32,
    gridY : u32,
    maxPerTile : u32,
};

@group(0) @binding(0) var<uniform> uGParams : GlobalParams;

@group(1) @binding(0) var<storage, read_write> outTileIndices : array<u32>;
@group(1) @binding(1) var<storage, read_write> outTileCounters : array<u32>;

@compute @workgroup_size(THREADS_PER_WORKGROUP)
fn cs_main(
    @builtin(local_invocation_id) lid : vec3<u32>,
    @builtin(workgroup_id) gid : vec3<u32>
) {
    let tileID = gid.x;
    let threadID = lid.x;

    if (threadID == 0u) {
        // thread 0 resets tile counter
        outTileCounters[tileID] = 0u;
    }

    for (var i = threadID; i < uGParams.maxPerTile; i += THREADS_PER_WORKGROUP) {
        // fill tile indices with sentinel value
        outTileIndices[tileID * uGParams.maxPerTile + i] = 0xFFFFFFFFu;
    }
}