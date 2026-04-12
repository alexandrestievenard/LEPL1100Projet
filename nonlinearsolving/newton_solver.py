from stiffness_non_linear import assemble_stiffness_and_rhs
from scipy.sparse import lil_matrix

import numpy as np

def assemble_residual(U, U_old, M, dt, elemTags, conn, jac, det, 
                      xphys, w, N, gN, kappa_fun, K_cap, 
                       r_growth, tag_to_dof, dirichlet_dofs=None, dirichlet_vals=None):
    
    ne = len(elemTags)
    ngp = len(w)
    nloc = int(len(conn) // ne)
    nn = int(np.max(tag_to_dof) + 1)

    det = np.asarray(det, dtype=np.float64).reshape(ne, ngp)
    xphys = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    jac = np.asarray(jac, dtype=np.float64).reshape(ne, ngp, 3, 3)
    conn = np.asarray(conn, dtype=np.int64).reshape(ne, nloc)
    N = np.asarray(N, dtype=np.float64).reshape(ngp, nloc)
    gN = np.asarray(gN, dtype=np.float64).reshape(ngp, nloc, 3)

    # Initialisation des 3 vecteurs résidu
    R1 = (M @ (U - U_old)) / dt
    R2 = np.zeros(nn)
    R3 = np.zeros(nn)

    for e in range(ne):
        element_tags = conn[e, :]
        dof_indices = tag_to_dof[element_tags]

        # Valeurs nodales sur l'élément e
        Ue = U[dof_indices]

        for g in range(ngp):
            xg = xphys[e, g]
            wg = w[g]
            detg = det[e, g]
            invjacg = np.linalg.inv(jac[e, g])

            # --------------------------------------------------
            # 1) Reconstruire u_g et grad_u_g au point de Gauss
            # --------------------------------------------------
            u_g = 0.0
            grad_u_g = np.zeros(gN.shape[2])

            for b in range(nloc):
                u_g += Ue[b] * N[g, b]
                gradNb = invjacg @ gN[g, b]
                grad_u_g += Ue[b] * gradNb

            # --------------------------------------------------
            # 2) Evaluer kappa au point de Gauss
            # --------------------------------------------------
            kappa_g = float(kappa_fun(u_g, xg))
            K_g = K_cap(xg)

            # --------------------------------------------------
            # 3) Ajouter la contribution au résidu diffusion
            # --------------------------------------------------
            for a in range(nloc):
                Na = N[g, a]
                gradNa = invjacg @ gN[g, a]
                Ia = int(dof_indices[a])

                R2[Ia] += wg * kappa_g * np.dot(grad_u_g, gradNa) * detg
                R3[Ia] += r_growth*u_g*(1-u_g/ K_g)*Na*wg*det[e,g]

    R = R1 + R2 - R3
    return R, R1, R2, R3