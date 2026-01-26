struct VertexOutput {
    @builtin(position) Position : vec4<f32>,
};

struct Vertex {
    pos : vec3<f32>,
    opacity : f32,
    cov1 : vec3<f32>,
    cov2 : vec3<f32>,
    color : vec3<f32>,
};

struct BinParams {
    vertexCount : u32,
    gridX : u32,
    gridY : u32,
    maxPerBin : u32,
};

struct CanvasParams {
    width : u32,
    height : u32,
};

@group(0) @binding(0) var<storage, read> tileVertices : array<Vertex>;
@group(0) @binding(1) var<storage, read> tileCounters : array<u32>;
@group(0) @binding(2) var<storage, read> params : BinParams;
@group(0) @binding(3) var<uniform> canvasParams : CanvasParams;

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(3.0, 1.0),
        vec2<f32>(-1.0, 1.0)
    );

    var output : VertexOutput;
    output.Position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
    return output;
}

@fragment
fn fs_main(@builtin(position) fragCoord : vec4<f32>) -> @location(0) vec4<f32> {
    // Normalize to [0,1] screen coordinates
    let uv = fragCoord.xy / vec2<f32>(f32(canvasParams.width), f32(canvasParams.height));

    // Compute which tile this fragment belongs to
    let tileX = u32(clamp(floor(uv.x * f32(params.gridX)), 0.0, f32(params.gridX - 1)));
    let tileY = u32(clamp(floor(uv.y * f32(params.gridY)), 0.0, f32(params.gridY - 1)));
    let tileIndex = u32(tileY * params.gridX + tileX);

    let aspect = f32(canvasParams.width) / f32(canvasParams.height);

    let count = tileCounters[tileIndex];
    var color = vec3<f32>(0.0);

    let r = 0.01; // splat radius in [0..1] screen space
    let sigma = r * 0.5; // adjust as needed

    for (var i = 0u; i < count; i = i + 1u) {
        let v = tileVertices[tileIndex * params.maxPerBin + i];

        // Compare fragment in global UV with vertex position in global UV
        // pos is in normalized clip space [-1,1]
        // uv is in [0,1]
        let vUV = (v.pos.xy + vec2<f32>(1.0)) * 0.5;
        let dx = abs(vUV.x - uv.x);
        let dy = abs(vUV.y - uv.y);

        // Scale dx by aspect ratio
        let dx_corrected = dx * aspect;

        let dist2 = dx_corrected * dx_corrected + dy * dy;
        let weight = exp(-dist2 / (2.0 * sigma * sigma));

        if (dist2 < r * r) {
            color += v.color * v.opacity * weight;
            color = min(color, vec3<f32>(1.0));
            if (color.x >= 1.0 && color.y >= 1.0 && color.z >= 1.0) { break; }
        }
    }

    // debug color: vertex count / maxPerBin
    // color = vec3<f32>(f32(count) / f32(params.maxPerBin));

    return vec4<f32>(color, 1.0);
    // return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
