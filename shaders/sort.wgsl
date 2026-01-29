const THREADS_PER_WORKGROUP = 256u;

struct Vertex {
    pos : vec3<f32>,
    opacity : f32,
    cov1 : vec3<f32>,
    cov2 : vec3<f32>,
    color : vec3<f32>,
};

struct TileParams {
    vertexCount : u32,
    gridX : u32,
    gridY : u32,
    maxPerTile : u32,
};

@group(0) @binding(0) var<storage, read> vertices : array<Vertex>;
@group(0) @binding(1) var<storage, read_write> tileIndices : array<u32>;
@group(0) @binding(2) var<storage, read> tileCounters : array<u32>;
@group(0) @binding(3) var<storage, read> params : TileParams;

var<workgroup> localIndices : array<u32, THREADS_PER_WORKGROUP>;
var<workgroup> localVertZs : array<f32, THREADS_PER_WORKGROUP>;

@compute @workgroup_size(THREADS_PER_WORKGROUP)
fn cs_main(@builtin(local_invocation_id) thread_local_id : vec3<u32>,
           @builtin(workgroup_id) workgroup_id : vec3<u32>) {

    let threadLocalID = thread_local_id.x;
    let tileID = workgroup_id.x;

    let idxCountInTile = tileCounters[tileID];
    // empty tile
    if (idxCountInTile == 0u) { return; }

    let baseIdx = tileID * params.maxPerTile;

    // load into shared memory from global memory
    for (var i = threadLocalID; i < idxCountInTile; i = i + THREADS_PER_WORKGROUP) {
        let vertexIdx = tileIndices[baseIdx + i];
        localIndices[i] = vertexIdx;
        localVertZs[i] = vertices[vertexIdx].pos.z;
    }

    // odd-even sort
    // must be done idxCountInTile iterations to guarantee sorted order
    for (var i = 0u; i < idxCountInTile - 1u; i = i + 1u) {

        let offset = i & 1u; // 0 for even, 1 for odd

        let leftIdx = threadLocalID * 2u + offset;
        let rightIdx = leftIdx + 1u;

        if (rightIdx < idxCountInTile) {
            let leftZ = localVertZs[leftIdx];
            let rightZ = localVertZs[rightIdx];

            // sort descending (far to near)
            if (leftZ < rightZ) {
                // swap
                let tempIdx = localIndices[leftIdx];
                localIndices[leftIdx] = localIndices[rightIdx];
                localIndices[rightIdx] = tempIdx;

                let tempZ = localVertZs[leftIdx];
                localVertZs[leftIdx] = localVertZs[rightIdx];
                localVertZs[rightIdx] = tempZ;
            }
        }

        // synchronize all threads before next iteration
        workgroupBarrier();
    }

    // write back to global memory
    for (var i = threadLocalID; i < idxCountInTile; i = i + THREADS_PER_WORKGROUP) {
        tileIndices[baseIdx + i] = localIndices[i];
    }
}