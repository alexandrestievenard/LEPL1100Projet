import numpy as np
from scipy.sparse import lil_matrix


def assemble_stiffness_and_rhs(elemTags, conn, jac, det, xphys, w, N, gN, U, kappa_fun, rhs_fun, tag_to_dof):  # ajout argument U
    """
    Assemble global stiffness matrix and load vector for:
        -d/dx (kappa(x) du/dx) = f(x)

    K_ij = ∫ kappa * grad(N_i)·grad(N_j) dx
    F_i  = ∫ f * N_i dx

    Notes:
    - gmsh gives gN in reference coordinates; we map with inv(J).
    - For 1D line embedded in 3D, gmsh provides a 3x3 Jacobian; we keep the same approach.

    Returns
    -------
    K : lil_matrix (nn x nn)
    F : ndarray (nn,)
    """
    ne   = len(elemTags)
    ngp  = len(w)                
    nloc = int(len(conn) // ne)  
    nn   = int(np.max(tag_to_dof) + 1) 

    det = np.asarray(det,dtype=np.float64).reshape(ne, ngp)
    xphys = np.asarray(xphys,dtype=np.float64).reshape(ne, ngp, 3)
    jac = np.asarray(jac,dtype=np.float64).reshape(ne, ngp, 3, 3)
    conn = np.asarray(conn,dtype=np.int64  ).reshape(ne, nloc)
    N = np.asarray(N,dtype=np.float64).reshape(ngp, nloc)
    gN = np.asarray(gN,dtype=np.float64).reshape(ngp, nloc, 3)

    K = lil_matrix((nn, nn), dtype=np.float64)
    F = np.zeros(nn, dtype=np.float64)

    for e in range(ne):
        element_tags = conn[e, :]
        dof_indices = tag_to_dof[element_tags]

        # Valeurs nodales de u^n restreintes à l'élément e
        # Ue[a] = valeur de la solution au nœud local a de l'élément e
        Ue = U[dof_indices]

        for g in range(ngp):
            xg = xphys[e, g]
            wg = w[g] 
            detg = det[e, g]  
            invjacg = np.linalg.inv(jac[e, g])

            # Reconstruction de u_h au point de Gauss par interpolation 
            u_g = float(np.dot(Ue, N[g, :]))

            # Évaluation de κ et f au point de Gauss
            kappa_g = float(kappa_fun(u_g, xg)) # kappa depend aussi de u_g
            f_g = float(rhs_fun(xg))

            for a in range(nloc):
                Ia = int(dof_indices[a])
                F[Ia] += wg * f_g * N[g, a] * detg
                
                gradNa = invjacg @ gN[g, a]
                for b in range(nloc):
                    Ib = int(dof_indices[b])
                    gradNb = invjacg @ gN[g, b]
                    K[Ia, Ib] += wg * kappa_g * float(np.dot(gradNa, gradNb)) * detg

    return K, F    

# plus besoin de assemble_rhs_neumann car servait a imposer un terme de bord neumann non homogene
#notre projet n'en utilise pas, car nos conditions de neumann sont homogenes sur les montagnes