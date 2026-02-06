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

@group(1) @binding(0) var<storage, read_write> inOutSplats : array<Splat2D>;

@compute @workgroup_size(THREADS_PER_WORKGROUP)
fn cs_main() {

}