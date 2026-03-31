# stiffness.py
import numpy as np
from scipy.sparse import lil_matrix


import numpy as np
from scipy.sparse import lil_matrix


def assemble_stiffness_and_rhs(nn, elemTags, conn, jac, det, xphys, w, N, gN, kappa_fun, rhs_fun):
    """
    Objects:
    --------
    coords_nodes : (nn, 3)
        Global node coordinates

    conn : (ne, nloc)
        conn[e, a] = global index of local node a in element e

    xi : (ngp, dim)
        Gauss points in reference element

    w : (ngp,)
        Quadrature weights

    N : (ngp, nloc)
        Basis functions at Gauss points
        N[g, a] = φ_a(ξ_g)

    gN : (ngp, nloc, dim)
        Gradients of basis functions in reference coords
        (NOT physical gradients)

    jac : (ne, ngp, 3, 3)
        Jacobian of mapping

    det : (ne, ngp)
        Determinant of Jacobian

    xphys : (ne, ngp, 3)
        Physical coordinates of Gauss points

    K[Ia, Ib] += ∫ kappa * grad φ_a · grad φ_b
    F[Ia]     += ∫ f * φ_a
    """
    ne = len(elemTags)
    ngp = len(w)

    conn = np.asarray(conn, dtype=np.int64)
    if conn.ndim == 1:
        nloc = len(conn) // ne
        conn = conn.reshape(ne, nloc)
    else:
        nloc = conn.shape[1]

    det = np.asarray(det, dtype=np.float64).reshape(ne, ngp)
    xphys = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    jac = np.asarray(jac, dtype=np.float64).reshape(ne, ngp, 3, 3)
    N = np.asarray(N, dtype=np.float64).reshape(ngp, nloc)
    gN = np.asarray(gN, dtype=np.float64).reshape(ngp, nloc, 3)

    K = lil_matrix((nn, nn), dtype=np.float64)
    F = np.zeros(nn, dtype=np.float64)

    for e in range(ne):
        nodes = conn[e, :] 
        for g in range(ngp):
            xg = xphys[e, g]
            wg = w[g]
            detg = det[e, g]
            invjacg = np.linalg.inv(jac[e, g])

            kappa_g = float(kappa_fun(xg))
            f_g = float(rhs_fun(xg))

            for a in range(nloc):
                Ia = int(nodes[a])
                F[Ia] += wg * f_g * N[g, a] * detg

                gradNa = invjacg.T @ gN[g, a]
                for b in range(nloc):
                    Ib = int(nodes[b])
                    gradNb = invjacg.T @ gN[g, b]
                    K[Ia, Ib] += wg * kappa_g * np.dot(gradNa, gradNb) * detg

    return K, F