struct Camera {
    vMatrix : mat4x4<f32>,
    pMatrix : mat4x4<f32>,
    pvMatrix : mat4x4<f32>,
};

struct RenderParams {
    scaleMultiplier : f32,
    showSfMPoints : f32, // 0.0 = false, 1.0 = true
};

struct CanvasParams {
    width : u32,
    height : u32,
};

struct Splat3D {
    pos : vec3<f32>,
    cov1 : vec3<f32>,
    cov2 : vec3<f32>,
    color : vec4<f32>,
};

// in Splat3D
// cov1 = (cxx, cxy, cxz)
// cov2 = (cyy, cyz, czz)
// therefore, 3x3 covariance is reconstructed as:
// [ cxx cxy cxz ]
// [ cxy cyy cyz ]
// [ cxz cyz czz ]

struct Splat2D {
    pos : vec3<f32>,
    cov : vec3<f32>,
    color : vec4<f32>,
};

// in Splat2D
// the transformed covariances in screen space is:
// [ cxx' cxy' ]
// [ cxy' cyy' ]
// where
// cov = (cxx', cxy', cyy')

@group(0) @binding(0) var<uniform> uCamera : Camera;
@group(0) @binding(1) var<uniform> uCanvasParams : CanvasParams;
@group(0) @binding(2) var<uniform> uRenderParams : RenderParams;

@group(1) @binding(0) var<storage, read> inSplats : array<Splat3D>;
@group(1) @binding(1) var<storage, read_write> outSplats : array<Splat2D>;
@group(1) @binding(2) var<storage, read_write> depthKeys : array<u32>;

@compute @workgroup_size(128)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&inSplats)) { return; }

    var s = inSplats[i];

    let pvM4x4 = uCamera.pvMatrix;

    /* ===== position transform ===== */
    let c = pvM4x4 * vec4<f32>(s.pos.xyz, 1.0);
    // perspective divide
    let transformedPos = vec3<f32>(
        c.x / c.w,
        c.y / c.w,
        c.z / c.w
    );

    /* ===== covariance transform ===== */
    let cov = mat3x3<f32>(
        s.cov1.x, s.cov1.y, s.cov1.z,
        s.cov1.y, s.cov2.x, s.cov2.y,
        s.cov1.z, s.cov2.y, s.cov2.z
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

    let cxx_p = dot(JR0, cov_JR0);
    let cxy_p = dot(JR0, cov_JR1);
    let cyy_p = dot(JR1, cov_JR1);

    var cxx_p_scaled : f32;
    var cxy_p_scaled : f32;
    var cyy_p_scaled : f32;
    var color: vec4<f32>;
    if (uRenderParams.showSfMPoints == 1.0) {
        let invAspect = f32(uCanvasParams.height) / f32(uCanvasParams.width);
        cxx_p_scaled = 2e-6 * invAspect;
        cxy_p_scaled = 0.0;
        cyy_p_scaled = 2e-6;
        color = vec4<f32>(s.color.rgb, 1.0);
    } else {
        let scaleMultiplier = uRenderParams.scaleMultiplier;
        let scaleMultiplierSq = scaleMultiplier * scaleMultiplier;
        cxx_p_scaled = cxx_p * scaleMultiplierSq;
        cxy_p_scaled = cxy_p * scaleMultiplierSq;
        cyy_p_scaled = cyy_p * scaleMultiplierSq;
        color = s.color;
    }

    outSplats[i] = Splat2D(
        transformedPos,
        vec3<f32>(cxx_p_scaled, cxy_p_scaled, cyy_p_scaled),
        color
    );

    depthKeys[i] = bitcast<u32>(transformedPos.z);
}
