// this sort requires each tile to hold 2^n indices.
// runs O((log^2)n)

// IMPLEMENTATION IS NOT CORRECT YET

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

@compute @workgroup_size(THREADS_PER_WORKGROUP)
fn cs_main(
    @builtin(local_invocation_id) lid : vec3<u32>,
    @builtin(workgroup_id) wid : vec3<u32>
) {
    let tid = lid.x;
    let tileID = wid.x;

    let count = min(tileCounters[tileID], params.maxPerTile);
    if (count <= 1u) { return; }

    let base = tileID * params.maxPerTile;
    let N = params.maxPerTile; // power-of-two

    // Bitonic sort (ascending by z)
    var k = 2u;
    loop {
        if (k > N) { break; }

        var j = k >> 1u;
        loop {
            if (j == 0u) { break; }

            // total comparisons = N / 2
            let totalPairs = N >> 1u;
            let pairsPerThread = (totalPairs + THREADS_PER_WORKGROUP - 1u) / THREADS_PER_WORKGROUP;

            for (var p = 0u; p < pairsPerThread; p++) {
                let pairIdx = p * THREADS_PER_WORKGROUP + tid;
                if (pairIdx >= totalPairs) { continue; }

                let i = pairIdx * 2u;
                let ixj = i ^ j;

                if (ixj > i) {
                    let ascending = ((i & k) == 0u);

                    let aIdx = base + i;
                    let bIdx = base + ixj;

                    let a = tileIndices[aIdx];
                    let b = tileIndices[bIdx];

                    // sentinel-safe
                    let za = select(
                        vertices[a].pos.z,
                        1e30,
                        a == 0xFFFFFFFFu
                    );
                    let zb = select(
                        vertices[b].pos.z,
                        1e30,
                        b == 0xFFFFFFFFu
                    );

                    let shouldSwap =
                        select(za > zb, za < zb, ascending);

                    if (shouldSwap) {
                        tileIndices[aIdx] = b;
                        tileIndices[bIdx] = a;
                    }
                }
            }

            workgroupBarrier();
            j >>= 1u;
        }

        k <<= 1u;
    }
}
