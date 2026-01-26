struct VertexOutput {
    @builtin(position) Position : vec4<f32>,
};

struct Vertex {
    pos : vec4<f32>, // already NDC [-1,1]
    cov1 : vec4<f32>,
    cov2 : vec4<f32>,
    opacity : f32,
    _pad : vec3<f32>,
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

@group(0) @binding(0) var<storage, read> binVertices : array<Vertex>;
@group(0) @binding(1) var<storage, read> binCounters : array<u32>;
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
    // Normalize to [0,1]
    let uv = fragCoord.xy / vec2<f32>(f32(canvasParams.width), f32(canvasParams.height));
    // Compute which bin this fragment belongs to
    let binX = u32(clamp(floor(uv.x * f32(params.gridX)), 0.0, f32(params.gridX - 1)));
    let binY = u32(clamp(floor(uv.y * f32(params.gridY)), 0.0, f32(params.gridY - 1)));
    let binIndex = binY * params.gridX + binX;

    let count = binCounters[binIndex];
    var color = vec3<f32>(0.0);

    for (var i = 0u; i < count; i = i + 1u) {
        let v = binVertices[binIndex * params.maxPerBin + i];

        // local fragment coordinates relative to bin (0..1 in bin)
        let binUV = uv * vec2<f32>(f32(params.gridX), f32(params.gridY)) - vec2<f32>(f32(binX), f32(binY));

        let dx = abs(v.pos.x - binUV.x);
        let dy = abs(v.pos.y - binUV.y);

        let r = 0.01; // splat radius
        if (dx < r && dy < r) {
            color += vec3<f32>(1.0, v.opacity, v.opacity) * v.opacity;
        }
    }

    // debug color: vertex count / maxPerBin
    color = vec3<f32>(f32(count) / f32(params.maxPerBin));

    return vec4<f32>(color, 1.0);
    // return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}