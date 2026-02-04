struct Camera {
    vMatrix : mat4x4<f32>,
    pMatrix : mat4x4<f32>,
    pvMatrix   : mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> uCamera : Camera;

struct VSIn {
    @location(0) pos : vec3<f32>,
};

struct VSOut {
    @builtin(position) clipPos : vec4<f32>,
    @location(0) color : vec4<f32>,
};

@vertex
fn vs_main(
    in : VSIn,
    @builtin(instance_index) instanceIdx : u32
) -> VSOut {
    var out : VSOut;

    let offset = vec2<f32>(
        select(-0.1, 0.1, (instanceIdx & 1u) == 1u),
        select(-0.05, 0.05, (instanceIdx & 2u) == 2u)
    );

    // position is world-space
    out.clipPos = uCamera.pvMatrix * vec4<f32>(in.pos, 1.0);

    out.clipPos.x += offset.x * out.clipPos.w;
    out.clipPos.y += offset.y * out.clipPos.w;

    // solid white
    out.color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    if (instanceIdx == 1u) {
        out.color = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    }
    if (instanceIdx == 2u) {
        out.color = vec4<f32>(0.0, 1.0, 0.0, 1.0);
    }
    if (instanceIdx == 3u) {
        out.color = vec4<f32>(0.0, 0.0, 1.0, 1.0);
    }
    return out;
}

@fragment
fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
    return in.color;
}
