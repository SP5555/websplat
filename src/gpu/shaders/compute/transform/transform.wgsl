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
    /*
        clip-space position is given by:
        PV * [x y z 1]^T = [c_x c_y c_z c_w]^T
        where:
        PV = [ PV00 PV01 PV02 PV03
               PV10 PV11 PV12 PV13
               PV20 PV21 PV22 PV23
               PV30 PV31 PV32 PV33 ]
        c_x = PV00*x + PV01*y + PV02*z + PV03*1
        c_y = PV10*x + PV11*y + PV12*z + PV13*1
        c_z = PV20*x + PV21*y + PV22*z + PV23*1
        c_w = PV30*x + PV31*y + PV32*z + PV33*1

        NDC is given by:
        [ X ] = [ c_x/c_w ]
        [ Y ]   [ c_y/c_w ]

        2x3 Jacobian of NDC [X Y] wrt. world position [x y z] is:
        J = [ dX/dx  dX/dy  dX/dz ]
            [ dY/dx  dY/dy  dY/dz ]
          = [ d(c_x/c_w)/dx  d(c_x/c_w)/dy  d(c_x/c_w)/dz ]
            [ d(c_y/c_w)/dx  d(c_y/c_w)/dy  d(c_y/c_w)/dz ]
          = [ (PV00*c_w-PV30*c_x)/c_w^2  (PV01*c_w-PV31*c_x)/c_w^2  (PV02*c_w-PV32*c_x)/c_w^2 ]
            [ (PV10*c_w-PV30*c_y)/c_w^2  (PV11*c_w-PV31*c_y)/c_w^2  (PV12*c_w-PV32*c_y)/c_w^2 ]
    */
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

    /* ===== J * W * cov * W^T * J^T (paper) version ===== */
    /*
        clip-space position is given by:
        V * [v_x v_y v_z 1]^T = [c_x c_y c_z c_w]^T
        where:
        V = [ V00 V01 V02 V03
              V10 V11 V12 V13
              V20 V21 V22 V23
              V30 V31 V32 V33 ]
        c_x = V00*v_x + V01*v_y + V02*v_z + V03*1
        c_y = V10*v_x + V11*v_y + V12*v_z + V13*1
        c_z = V20*v_x + V21*v_y + V22*v_z + V23*1
        c_w = V30*v_x + V31*v_y + V32*v_z + V33*1

        NDC is given by:
        [ X ] = [ c_x/c_w ]
        [ Y ]   [ c_y/c_w ]

        2x3 Jacobian of NDC [X Y] wrt. world position [v_x v_y v_z] is:
        J = [ dX/dv_x  dX/dv_y  dX/dv_z ]
            [ dY/dv_x  dY/dv_y  dY/dv_z ]
          = [ d(c_x/c_w)/dv_x  d(c_x/c_w)/dv_y  d(c_x/c_w)/dv_z ]
            [ d(c_y/c_w)/dv_x  d(c_y/c_w)/dv_y  d(c_y/c_w)/dv_z ]
          = [ (V00*c_w-V30*c_x)/c_w^2  (V01*c_w-V31*c_x)/c_w^2  (V02*c_w-V32*c_x)/c_w^2 ]
            [ (V10*c_w-V30*c_y)/c_w^2  (V11*c_w-V31*c_y)/c_w^2  (V12*c_w-V32*c_y)/c_w^2 ]
    */
    // let W = mat3x3<f32>(
    //     uCamera.vMatrix[0].xyz,
    //     uCamera.vMatrix[1].xyz,
    //     uCamera.vMatrix[2].xyz
    // );
    // let W_cov_WT = W * cov * transpose(W);
    // let J = mat3x2<f32>(
    //     vec2<f32>(
    //         (uCamera.pMatrix[0][0] * c.w - uCamera.pMatrix[0][3] * c.x) * inv_c_w2,
    //         (uCamera.pMatrix[0][1] * c.w - uCamera.pMatrix[0][3] * c.y) * inv_c_w2
    //     ),
    //     vec2<f32>(
    //         (uCamera.pMatrix[1][0] * c.w - uCamera.pMatrix[1][3] * c.x) * inv_c_w2,
    //         (uCamera.pMatrix[1][1] * c.w - uCamera.pMatrix[1][3] * c.y) * inv_c_w2,
    //     ),
    //     vec2<f32>(
    //         (uCamera.pMatrix[2][0] * c.w - uCamera.pMatrix[2][3] * c.x) * inv_c_w2,
    //         (uCamera.pMatrix[2][1] * c.w - uCamera.pMatrix[2][3] * c.y) * inv_c_w2,
    //     )
    // );
    // let J_W_cov_WT_JT = J * W_cov_WT * transpose(J);

    // let cxx_p = J_W_cov_WT_JT[0][0];
    // let cxy_p = J_W_cov_WT_JT[0][1];
    // let cyy_p = J_W_cov_WT_JT[1][1];

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
}
