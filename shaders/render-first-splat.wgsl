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

struct TileParams {
    vertexCount : u32,
    gridX : u32,
    gridY : u32,
    maxPerTile : u32,
};

struct CanvasParams {
    width : u32,
    height : u32,
};

@group(0) @binding(0) var<storage, read> vertices : array<Vertex>;
@group(0) @binding(1) var<storage, read> tileIndices : array<u32>;
@group(0) @binding(2) var<storage, read> tileCounters : array<u32>;
@group(0) @binding(3) var<storage, read> params : TileParams;
@group(0) @binding(4) var<uniform> canvasParams : CanvasParams;

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
    let uv = fragCoord.xy / vec2<f32>(f32(canvasParams.width), f32(canvasParams.height));
    let tileX = u32(clamp(floor(uv.x * f32(params.gridX)), 0.0, f32(params.gridX - 1)));
    let tileY = u32(clamp(floor(uv.y * f32(params.gridY)), 0.0, f32(params.gridY - 1)));
    let tileIndex = u32(tileY * params.gridX + tileX);

    let count = tileCounters[tileIndex];
    let fragNDC = uv * 2.0 - vec2<f32>(1.0);

    for (var i = 0u; i < count; i = i + 1u) {
        let v = vertices[tileIndices[tileIndex * params.maxPerTile + i]];

        let dx = fragNDC.x - v.pos.x;
        let dy = fragNDC.y - v.pos.y;

        let cxx = v.cov1.x;
        let cxy = v.cov1.y;
        let cyy = v.cov1.z;

        let det = cxx * cyy - cxy * cxy;
        if (det <= 0.0) {
            continue;
        }
        let invDet = 1.0 / det;
        let invCxx =  cyy * invDet;
        let invCxy = -cxy * invDet;
        let invCyy =  cxx * invDet;

        let dist2 = dx * dx * invCxx + 2.0 * dx * dy * invCxy + dy * dy * invCyy;

        if (dist2 < 9.0) {
            // first splat found, just return its color
            return vec4<f32>(v.color, 1.0);
        }
    }

    // background if no splat
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
