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

@group(1) @binding(0) var<storage, read_write> inSplats : array<Splat2D>;
@group(1) @binding(1) var<storage, read> outSplats : array<Splat2D>;
@group(1) @binding(2) var<storage, read_write> depthKeys : array<u32>;

fn f32_to_sortable_u32(x: f32) -> u32 {
    let b = bitcast<u32>(x);

    // positive floats: flip sign bit
    // negative floats: invert all bits
    return select(~b, b ^ 0x80000000u, (b >> 31u) == 0u);
}

// copy outSplats to inSplats
@compute @workgroup_size(THREADS_PER_WORKGROUP)
fn cs_main(
    @builtin(global_invocation_id) global_id : vec3<u32>
) {
    let index = global_id.x;

    if (index < uSceneParams.splatCount) {
        let splat =  outSplats[index];
        inSplats[index] = splat;

        // update depth key
        depthKeys[index] = f32_to_sortable_u32(splat.pos.z);
    }
}