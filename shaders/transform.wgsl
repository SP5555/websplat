struct Camera {
    vMatrix : mat4x4<f32>,
    pMatrix : mat4x4<f32>,
};

struct Vertex {
    pos : vec4<f32>,    // vec3
    cov1 : vec4<f32>,   // vec3
    cov2 : vec4<f32>,   // vec3
    opacity : f32,
    _pad : vec3<f32>,
};

@group(0) @binding(0) var<uniform> camera : Camera;
@group(0) @binding(1) var<storage, read> vertices : array<Vertex>;
@group(0) @binding(2) var<storage, read_write> outVertices : array<Vertex>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&vertices)) {
        return;
    }

    let v = vertices[i];

    let t = camera.pMatrix * camera.vMatrix * vec4<f32>(v.pos.xyz, 1.0);
    // perspective divide
    let transformedPos = vec4<f32>(t.xyz / t.w, 1.0);

    outVertices[i] = Vertex(transformedPos, v.cov1, v.cov2, v.opacity, v._pad);
}
