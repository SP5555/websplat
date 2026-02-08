@group(0) @binding(0) var<storage, read_write> tileCounters : array<u32>;

// only one workgroup is launched
@compute @workgroup_size(1)
fn cs_main() {
    var sum = 0u;
    for (var i = 0u; i < arrayLength(&tileCounters); i = i + 1u) {
        tileCounters[i] = sum;
        sum = sum + tileCounters[i];
    }
}