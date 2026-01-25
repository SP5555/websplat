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

@group(0) @binding(0) var<storage, read> binVertices : array<Vertex>;
@group(0) @binding(1) var<storage, read> binCounters : array<u32>;
@group(0) @binding(2) var<uniform> params : BinParams;

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
    // convert fragCoord.xy to NDC [-1,1]
    let uv = fragCoord.xy / vec2<f32>(f32(params.gridX * 16), f32(params.gridY * 16)) * 2.0 - 1.0;

    var color = vec3<f32>(0.0);

    for (var binY = 0u; binY < params.gridY; binY = binY + 1u) {
        for (var binX = 0u; binX < params.gridX; binX = binX + 1u) {
            let binIndex = binY * params.gridX + binX;
            let count = binCounters[binIndex];

            for (var i = 0u; i < count; i = i + 1u) {
                let v = binVertices[binIndex * params.maxPerBin + i];

                // screen-space position
                let dx = abs(v.pos.x - uv.x);
                let dy = abs(v.pos.y - uv.y);
                let r = 0.01; // splat radius
                if (dx < r && dy < r) {
                    color += vec3<f32>(1.0, v.opacity, v.opacity) * v.opacity;
                }
            }
        }
    }

    return vec4<f32>(color, 1.0);
}
