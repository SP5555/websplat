struct Camera {
    vMatrix : mat4x4<f32>,
    pMatrix : mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera : Camera;

struct VertexInput {
    @location(0) position : vec3<f32>,
    @location(1) opacity : f32,
    @location(2) covariance1 : vec3<f32>,
    @location(3) covariance2 : vec3<f32>,
    @location(4) color : vec3<f32>,
};

struct VertexOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) fragOpacity : f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var output : VertexOutput;
    output.Position = camera.pMatrix * camera.vMatrix * vec4<f32>(in.position.xyz, 1.0);
    output.fragOpacity = in.opacity;
    return output;
}

struct FragmentInput {
    @location(0) fragOpacity : f32,
};

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, in.fragOpacity, in.fragOpacity, 1.0);
}
