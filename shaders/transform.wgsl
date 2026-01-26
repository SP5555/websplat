struct Camera {
    vMatrix : mat4x4<f32>,
    pMatrix : mat4x4<f32>,
};

struct Vertex {
    pos : vec3<f32>,
    opacity : f32,
    cov1 : vec3<f32>,
    cov2 : vec3<f32>,
};

@group(0) @binding(0) var<uniform> camera : Camera;
@group(0) @binding(1) var<storage, read> vertices : array<Vertex>;
@group(0) @binding(2) var<storage, read_write> outVertices : array<Vertex>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&vertices)) { return; }

    let v = vertices[i];

    let t = camera.pMatrix * camera.vMatrix * vec4<f32>(v.pos.xyz, 1.0);
    // perspective divide
    let transformedPos = vec3<f32>(
        t.x / t.w,
        - t.y / t.w,
        t.z / t.w
    );

    outVertices[i] = Vertex(transformedPos, v.opacity, v.cov1, v.cov2);
}
