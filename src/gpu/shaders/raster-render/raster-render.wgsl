struct Camera {
    vMatrix : mat4x4<f32>,
    pMatrix : mat4x4<f32>,
    pvMatrix : mat4x4<f32>,
};

struct CanvasParams {
    width : u32,
    height : u32,
};

struct Splat {
    pos : vec3<f32>,
    cov1 : vec3<f32>,
    cov2 : vec3<f32>,
    color : vec4<f32>,
};

@group(0) @binding(0) var<uniform> uCamera : Camera;
@group(0) @binding(1) var<uniform> uCParams : CanvasParams;

@group(1) @binding(0) var<storage, read> inSplats : array<Splat>;

struct VSOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) splatPos : vec3<f32>,
    @location(1) color : vec4<f32>,
    @location(2) cov : vec3<f32>, // cxx', cxy', cyy'
};

@vertex
fn vs_main(
    @builtin(vertex_index) quadIdx : u32,
    @builtin(instance_index) splatIdx : u32
) -> VSOut {

    let quadOffsetDirs = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0),
        vec2( 1.0, -1.0),
        vec2(-1.0,  1.0),

        vec2(-1.0,  1.0),
        vec2( 1.0, -1.0),
        vec2( 1.0,  1.0),
    );

    let splat = inSplats[splatIdx];
    let offsetDir = quadOffsetDirs[quadIdx];

    let pvM4x4 = uCamera.pvMatrix;

    /* ===== position transform ===== */
    let c = pvM4x4 * vec4<f32>(splat.pos, 1.0);
    // perspective divide
    let ndcPos = vec3<f32>(
        c.x / c.w,
        c.y / c.w,
        c.z / c.w
    );

    /* ===== covariance transform ===== */
    let cov = mat3x3<f32>(
        splat.cov1.x, splat.cov1.y, splat.cov1.z,
        splat.cov1.y, splat.cov2.x, splat.cov2.y,
        splat.cov1.z, splat.cov2.y, splat.cov2.z
    );
    let w = max(c.w, 1e-6);
    let inv_c_w2 = 1.0 / (w * w);

    /* ===== J * cov * J^T (feelings) version ===== */
    // Jacobian
    // note that pvM4x4 is column-major
    let JR0 = vec3<f32>(
        (pvM4x4[0][0] * c.w - pvM4x4[0][3] * c.x) * inv_c_w2,
        (pvM4x4[1][0] * c.w - pvM4x4[1][3] * c.x) * inv_c_w2,
        (pvM4x4[2][0] * c.w - pvM4x4[2][3] * c.x) * inv_c_w2
    );
    let JR1 = vec3<f32>(
        (pvM4x4[0][1] * c.w - pvM4x4[0][3] * c.y) * inv_c_w2,
        (pvM4x4[1][1] * c.w - pvM4x4[1][3] * c.y) * inv_c_w2,
        (pvM4x4[2][1] * c.w - pvM4x4[2][3] * c.y) * inv_c_w2
    );

    // transformed covariance = J * cov * J^T
    let cov_JR0 = cov * JR0;
    let cov_JR1 = cov * JR1;

    let cxx_ndc = dot(JR0, cov_JR0);
    let cxy_ndc = dot(JR0, cov_JR1);
    let cyy_ndc = dot(JR1, cov_JR1);

    // pixel space covariance for bounding box calculation
    let sx = f32(uCParams.width) * 0.5;
    let sy = f32(uCParams.height) * 0.5;
    let cxx_p = cxx_ndc * sx * sx;
    let cxy_p = cxy_ndc * sx * sy;
    let cyy_p = cyy_ndc * sy * sy;

    let trace = cxx_p + cyy_p;
    let det = cxx_p * cyy_p - cxy_p * cxy_p;
    
    let temp = max(trace * trace - 4.0 * det, 0.0);
    let lambda1 = 0.5 * (trace + sqrt(temp));
    let lambda2 = 0.5 * (trace - sqrt(temp));

    // 3 sigma max radius in pixel space
    let maxRadius_p = 3.0 * sqrt(max(lambda1, lambda2));

    let X = ndcPos.x + offsetDir.x * maxRadius_p / sx;
    let Y = ndcPos.y + offsetDir.y * maxRadius_p / sy;

    return VSOut(
        vec4<f32>(X * c.w, Y * c.w, c.z, c.w),
        ndcPos,
        splat.color,
        vec3<f32>(cxx_ndc, cxy_ndc, cyy_ndc)
    );
}

@fragment
fn fs_main(  
    in : VSOut
) -> @location(0) vec4<f32> {
    // Normalize to [-1,1] screen coordinates
    // in.pos is now fragCoord somehow???
    // fragCoord.y grows upwards, so we need to flip it
    let fragNDC = vec2<f32>(
        in.pos.x / f32(uCParams.width) * 2.0 - 1.0,
        (1.0 - in.pos.y / f32(uCParams.height)) * 2.0 - 1.0
    );
    let centerNDC = in.splatPos.xy;

    // in NDC space
    let dx = fragNDC.x - centerNDC.x;
    let dy = fragNDC.y - centerNDC.y;

    let cxx = in.cov.x;
    let cxy = in.cov.y;
    let cyy = in.cov.z;

    let det = cxx * cyy - cxy * cxy;
    if (det <= 0.0) {
        discard; // broken, skip
    }

    let invDet = 1.0 / det;

    let invCxx =  cyy * invDet;
    let invCxy = -cxy * invDet;
    let invCyy =  cxx * invDet;

    // Mahalanobis distance squared
    let dist2 = dx * dx * invCxx + 2.0 * dx * dy * invCxy + dy * dy * invCyy;

    // return vec4<f32>(1.0, 1.0, 1.0, 0.04);

    if (dist2 > 9.0) { // 3 sigma
        discard;
    }

    let weight = exp(-dist2 * 0.5);
    let alpha = clamp(in.color.a * weight, 0.0, 1.0);

    return vec4<f32>(in.color.rgb, alpha);
}
