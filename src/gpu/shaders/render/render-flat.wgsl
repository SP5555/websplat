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

@group(1) @binding(0) var<storage, read> inVertices : array<Vertex>;
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
    let uv = fragCoord.xy / vec2<f32>(f32(uCParams.width), f32(uCParams.height));
    let tileX = u32(clamp(floor(uv.x * f32(uGParams.gridX)), 0.0, f32(uGParams.gridX - 1)));
    let tileY = u32(clamp(floor(uv.y * f32(uGParams.gridY)), 0.0, f32(uGParams.gridY - 1)));
    let tileID = u32(tileY * uGParams.gridX + tileX);

    let count = min(inTileCounters[tileID], uGParams.maxPerTile);

    var accumColor = vec3<f32>(0.0);
    var accumAlpha = 0.0;

    // uv is in [0,1], fragNDC is in [-1,1]
    let fragNDC = uv * 2.0 - vec2<f32>(1.0);

    for (var i = 0u; i < count; i = i + 1u) {
        let v = inVertices[inTileIndices[tileID * uGParams.maxPerTile + i]];

        let dx = fragNDC.x - v.pos.x;
        let dy = fragNDC.y - v.pos.y;

        let cxx = v.cov1.x;
        let cxy = v.cov1.y;
        let cyy = v.cov1.z;

        let det = cxx * cyy - cxy * cxy;
        if (det <= 0.0) {
            continue; // broken, skip
        }
        let invDet = 1.0 / det;

        let invCxx =  cyy * invDet;
        let invCxy = -cxy * invDet;
        let invCyy =  cxx * invDet;

        // Mahalanobis distance squared
        let dist2 = dx * dx * invCxx + 2.0 * dx * dy * invCxy + dy * dy * invCyy;

        if (dist2 < 9.0) { // 3 sigma
            let alpha = clamp(v.opacity, 0.0, 1.0);

            accumColor += (1.0 - accumAlpha) * v.color * alpha;
            accumAlpha += (1.0 - accumAlpha) * alpha;

            // very opaque, stop early
            if (accumAlpha >= 0.99) {
                break;
            }
        }
    }

    return vec4<f32>(accumColor, 1.0);
}
