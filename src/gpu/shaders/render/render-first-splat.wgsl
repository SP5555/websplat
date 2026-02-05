struct VertexOutput {
    @builtin(position) Position : vec4<f32>,
};

struct Splat2D {
    pos : vec3<f32>,
    cov : vec3<f32>,
    color : vec4<f32>,
};

struct GlobalParams {
    vertexCount : u32,
    gridX : u32,
    gridY : u32,
    maxPerTile : u32,
};

struct CanvasParams {
    width : u32,
    height : u32,
};

@group(0) @binding(0) var<uniform> uGParams : GlobalParams;
@group(0) @binding(1) var<uniform> uCParams : CanvasParams;

@group(1) @binding(0) var<storage, read> inSplats : array<Splat2D>;
@group(1) @binding(1) var<storage, read> inTileIndices : array<u32>;
@group(1) @binding(2) var<storage, read> inTileCounters : array<u32>;

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
    let uv = vec2<f32>(
        fragCoord.x / f32(uCParams.width),
        1.0 - fragCoord.y / f32(uCParams.height)
    );
    let tileX = u32(clamp(floor(uv.x * f32(uGParams.gridX)), 0.0, f32(uGParams.gridX - 1)));
    let tileY = u32(clamp(floor(uv.y * f32(uGParams.gridY)), 0.0, f32(uGParams.gridY - 1)));
    let tileIndex = u32(tileY * uGParams.gridX + tileX);

    let count = inTileCounters[tileIndex];
    let fragNDC = uv * 2.0 - vec2<f32>(1.0);

    for (var i = 0u; i < count; i = i + 1u) {
        let s = inSplats[inTileIndices[tileIndex * uGParams.maxPerTile + i]];

        let dx = fragNDC.x - s.pos.x;
        let dy = fragNDC.y - s.pos.y;

        let cxx = s.cov.x;
        let cxy = s.cov.y;
        let cyy = s.cov.z;

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
            return vec4<f32>(s.color.rgb, s.color.a);
        }
    }

    // background if no splat
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
