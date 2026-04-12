from stiffness_non_linear import assemble_stiffness_and_rhs
from scipy.sparse import lil_matrix

import numpy as np
from scipy.sparse.linalg import spsolve

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

    if dirichlet_dofs is not None and dirichlet_vals is not None:
        R[dirichlet_dofs] = U[dirichlet_dofs] - dirichlet_vals
    
    return R

def assemble_jacobian(U, M, dt, elemTags, conn, jac, det,
                      xphys, w, N, gN, kappa_fun, dkappa_du, K_cap,
                      r_growth, tag_to_dof, dirichlet_dofs=None):

    ne = len(elemTags)
    ngp = len(w)
    nloc = int(len(conn) // ne)
    nn = int(np.max(tag_to_dof) + 1)

    det   = np.asarray(det, dtype=np.float64).reshape(ne, ngp)
    xphys = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    jac   = np.asarray(jac, dtype=np.float64).reshape(ne, ngp, 3, 3)
    conn  = np.asarray(conn, dtype=np.int64).reshape(ne, nloc)
    N     = np.asarray(N, dtype=np.float64).reshape(ngp, nloc)
    gN    = np.asarray(gN, dtype=np.float64).reshape(ngp, nloc, 3)

    J1 = (M / dt).tolil()
    J2 = lil_matrix((nn, nn), dtype=np.float64)
    J3 = lil_matrix((nn, nn), dtype=np.float64)

    for e in range(ne):
        element_tags = conn[e, :]
        dof_indices = tag_to_dof[element_tags]
        Ue = U[dof_indices]

        for g in range(ngp):
            xg = xphys[e, g]
            wg = w[g]
            detg = det[e, g]
            invjacg = np.linalg.inv(jac[e, g])

            u_g = 0.0
            grad_u_g = np.zeros(gN.shape[2])

            for b in range(nloc):
                u_g += Ue[b] * N[g, b]
                gradNb = invjacg @ gN[g, b]
                grad_u_g += Ue[b] * gradNb

            K_g = float(K_cap(xg))
            kappa_g = float(kappa_fun(u_g, xg))
            dkappa_g = float(dkappa_du(u_g, xg))
            df_du = r_growth * (1.0 - 2.0 * u_g / K_g)

            for b in range(nloc):
                Nb = N[g, b]
                Ib = int(dof_indices[b])
                gradNb = invjacg @ gN[g, b]

                for a in range(nloc):
                    Na = N[g, a]
                    Ia = int(dof_indices[a])
                    gradNa = invjacg @ gN[g, a]

                    term_lin = kappa_g * np.dot(gradNb, gradNa)
                    term_nonlin = dkappa_g * Nb * np.dot(grad_u_g, gradNa)

                    J2[Ia, Ib] += (term_lin + term_nonlin) * wg * detg
                    J3[Ia, Ib] += df_du * Nb * Na * wg * detg

    J = J1 + J2 - J3
    J = J.tolil()

    if dirichlet_dofs is not None:
        for i in dirichlet_dofs:
            J.rows[i] = [i]
            J.data[i] = [1.0]

        # annuler aussi les colonnes Dirichlet
        dir_set = set(int(i) for i in dirichlet_dofs)
        for row in range(J.shape[0]):
            if row in dir_set:
                continue
            new_cols = []
            new_vals = []
            for col, val in zip(J.rows[row], J.data[row]):
                if col not in dir_set:
                    new_cols.append(col)
                    new_vals.append(val)
            J.rows[row] = new_cols
            J.data[row] = new_vals

    return J.tocsr()

def newton_solver(
    U_init,
    U_old,
    M, dt,
    elemTags, conn, jac, det, xphys, w, N, gN,
    kappa_fun, dkappa_du, K_cap, r_growth,
    tag_to_dof,
    dirichlet_dofs=None, dirichlet_vals=None,
    tol=1e-8, max_iter=20):

    U = U_init.copy()

    for k in range(max_iter):
        
        # 1) Calcul du résidu
        R = assemble_residual(
            U, U_old, M, dt,
            elemTags, conn, jac, det, xphys, w, N, gN,
            kappa_fun, K_cap, r_growth,
            tag_to_dof,
            dirichlet_dofs, dirichlet_vals
        )

        norm_R = np.linalg.norm(R)

        # critère d'arrêt
        if norm_R < tol:
            break

        # 2) Jacobienne
        J = assemble_jacobian(
            U, M, dt,
            elemTags, conn, jac, det, xphys, w, N, gN,
            kappa_fun, dkappa_du, K_cap, r_growth,
            tag_to_dof,
            dirichlet_dofs
        )

        # 3) Résolution du système linéaire
        deltaU = spsolve(J, -R)

        # 4) Mise à jour
        U += deltaU

        # sécurité (optionnel mais utile)
        if np.linalg.norm(deltaU) < tol:
            break

    return U
    