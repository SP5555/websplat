struct Camera {
    vMatrix : mat4x4<f32>,
    pMatrix : mat4x4<f32>,
};

struct Vertex {
    pos : vec3<f32>,
    opacity : f32,
    cov1 : vec3<f32>,
    cov2 : vec3<f32>,
    color : vec3<f32>,
};

// in vertices
// cov1 = (cxx, cxy, cxz)
// cov2 = (cyy, cyz, czz)
// therefore, 3x3 covariance is reconstructed as:
// [ cxx cxy cxz ]
// [ cxy cyy cyz ]
// [ cxz cyz czz ]

// in outVertices
// the transformed covariances in screen space is:
// [ cxx' cxy'  0 ]
// [ cxy' cyy'  0 ]
// [   0    0   1 ]
// where
// cov1 = (cxx', cxy', cyy')

@group(0) @binding(0) var<uniform> camera : Camera;
@group(0) @binding(1) var<storage, read> vertices : array<Vertex>;
@group(0) @binding(2) var<storage, read_write> outVertices : array<Vertex>;

@compute @workgroup_size(128)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&vertices)) { return; }

    let v = vertices[i];

    let pvMatrix = camera.pMatrix * camera.vMatrix;

    /* ===== position transform ===== */
    let c = pvMatrix * vec4<f32>(v.pos.xyz, 1.0);
    // perspective divide
    let transformedPos = vec3<f32>(
        c.x / c.w,
        - c.y / c.w,
        c.z / c.w
    );

    /* ===== covariance transform ===== */
    /*
        clip-space position is given by:
        PV * [x y z 1]^T = [c_x c_y c_z c_w]^T

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

    let cov = mat3x3<f32>(
        v.cov1.x, v.cov1.y, v.cov1.z,
        v.cov1.y, v.cov2.x, v.cov2.y,
        v.cov1.z, v.cov2.y, v.cov2.z
    );

    let inv_c_w2 = 1.0 / (c.w * c.w);

    // Jacobian
    let J0 = vec3<f32>(
        (pvMatrix[0][0] * c.w - pvMatrix[3][0] * c.x) * inv_c_w2,
        (pvMatrix[0][1] * c.w - pvMatrix[3][1] * c.x) * inv_c_w2,
        (pvMatrix[0][2] * c.w - pvMatrix[3][2] * c.x) * inv_c_w2
    );
    let J1 = vec3<f32>(
        (pvMatrix[1][0] * c.w - pvMatrix[3][0] * c.y) * inv_c_w2,
        (pvMatrix[1][1] * c.w - pvMatrix[3][1] * c.y) * inv_c_w2,
        (pvMatrix[1][2] * c.w - pvMatrix[3][2] * c.y) * inv_c_w2
    );

    // transformed covariance = J * cov * J^T
    let cov_J0 = cov * J0;
    let cov_J1 = cov * J1;

    let cxx_p = dot(J0, cov_J0);
    let cxy_p = dot(J0, cov_J1);
    let cyy_p = dot(J1, cov_J1);

    outVertices[i] = Vertex(
        transformedPos,
        v.opacity,
        vec3<f32>(cxx_p, cxy_p, cyy_p),
        v.cov2,
        v.color
    );
}
