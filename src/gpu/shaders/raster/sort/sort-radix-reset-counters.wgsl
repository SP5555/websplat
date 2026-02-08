@group(1) @binding(0) var<storage, read_write> radixGlobalCounters : array<atomic<u32>>;

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let idx = global_id.x;
    atomicStore(&radixGlobalCounters[idx], 0u);
}