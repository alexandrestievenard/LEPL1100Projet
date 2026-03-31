# mass.py
import numpy as np
from scipy.sparse import lil_matrix


def assemble_mass(elemTags, conn, det, w, N):
    """
    Assemble global mass matrix:
        M_ij = sum_e ∫_e N_i N_j dx

    Parameters
    ----------
    elemTags : array-like, shape (ne,)
    conn     : flattened connectivity (ne*nloc)
    det      : flattened det(J) values (ne*ngp)
    w        : quadrature weights (ngp)
    N        : flattened basis values (ngp*nloc)

    Returns
    -------
    M : lil_matrix (nn x nn)
    """
    ne = len(elemTags)
    ngp = len(w)
    nloc = int(len(conn) // ne)
    nn = int(np.max(conn))  # gmsh node tags are 1..nn (assuming compact)

    det = np.asarray(det, dtype=np.float64).reshape(ne, ngp)
    conn = np.asarray(conn, dtype=np.int64).reshape(ne, nloc)
    N = np.asarray(N, dtype=np.float64).reshape(ngp, nloc)

    M = lil_matrix((nn, nn), dtype=np.float64)

    for e in range(ne):
        nodes = conn[e, :] - 1  # to 0-based dof ids
        for g in range(ngp):
            wg = w[g]
            detg = det[e, g]
            for a in range(nloc):
                Ia = int(nodes[a])
                Na = N[g, a]
                for b in range(nloc):
                    Ib = int(nodes[b])
                    M[Ia, Ib] += wg * Na * N[g, b] * detg

    return M
