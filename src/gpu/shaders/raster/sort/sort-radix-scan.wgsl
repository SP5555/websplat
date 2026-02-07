@group(1) @binding(0) var<storage, read_write> radixGlobalCounters : array<atomic<u32>>;

// only one workgroup, one thread is launched
@compute @workgroup_size(1)
fn cs_main() {
    var accum: u32 = 0u;
    // perform exclusive scan on 256 global counters
    for (var i: u32 = 0u; i < 256u; i = i + 1u) {
        let count = atomicLoad(&radixGlobalCounters[i]);
        atomicStore(&radixGlobalCounters[i], accum);
        accum = accum + count;
    }
}