from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import numpy as np


def preprocess_newton_data(elemTags, conn, jac, det, xphys, w, N, gN, tag_to_dof):
    """
    Pré-calcul des données éléments finis utilisées dans Newton.

    Cette fonction rassemble tous les objets FEM qui ne dépendent
    ni du temps, ni de l'itération de Newton. L'idée est simple :
    au lieu de refaire les mêmes opérations coûteuses à chaque appel
    de assemble_residual() et assemble_jacobian(), on les fait une
    seule fois ici.

    On pré-calcule notamment :
    - les tableaux reshaped (connectivité, jacobiens, poids, etc.)
    - les indices globaux de DDL pour chaque élément
    - les inverses des jacobiens aux points de Gauss
    - les gradients physiques des fonctions de forme

    Paramètres
    ----------
    elemTags : array-like, shape (ne,)
        Tags des éléments.
    conn : array-like
        Connectivité des éléments en tags Gmsh.
        Peut être déjà de forme (ne, nloc) ou aplatie.
    jac : array-like
        Jacobien géométrique aux points de Gauss.
        Peut être aplati ou déjà de forme (ne, ngp, 3, 3).
    det : array-like
        Déterminant du jacobien aux points de Gauss.
    xphys : array-like
        Coordonnées physiques des points de Gauss.
    w : array-like, shape (ngp,)
        Poids de quadrature.
    N : array-like
        Valeurs des fonctions de forme aux points de Gauss.
    gN : array-like
        Gradients des fonctions de forme dans l'élément de référence.
    tag_to_dof : ndarray
        Tableau de conversion "tag Gmsh -> indice compact de DDL".

    Retour
    ------
    data : dict
        Dictionnaire contenant toutes les données pré-calculées
        utiles pour l'assemblage du résidu et de la jacobienne.
    """

    # ============================================================
    # 1) Dimensions globales du problème
    # ============================================================
    ne = len(elemTags)   # nombre d'éléments
    ngp = len(w)         # nombre de points de Gauss par élément

    # ============================================================
    # 2) Mise en forme cohérente de la connectivité
    # ============================================================
    # On convertit la connectivité en tableau numpy d'entiers.
    # Si le tableau est aplati, on le remet sous la forme (ne, nloc),
    # où nloc = nombre de DDL locaux par élément.
    conn = np.asarray(conn, dtype=np.int64)

    if conn.ndim == 1:
        nloc = conn.size // ne
        conn = conn.reshape(ne, nloc)
    else:
        nloc = conn.shape[1]

    # ============================================================
    # 3) Reshape des objets FEM et géométriques
    # ============================================================
    det   = np.asarray(det,   dtype=np.float64).reshape(ne, ngp)
    xphys = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    jac   = np.asarray(jac,   dtype=np.float64).reshape(ne, ngp, 3, 3)
    N     = np.asarray(N,     dtype=np.float64).reshape(ngp, nloc)
    gN    = np.asarray(gN,    dtype=np.float64).reshape(ngp, nloc, 3)
    w     = np.asarray(w,     dtype=np.float64)

    # ============================================================
    # 4) Conversion tags Gmsh -> indices compacts de DDL
    # ============================================================
    dof_indices = tag_to_dof[conn]   # shape = (ne, nloc)

    # ============================================================
    # 5) Inversion des jacobiens une seule fois
    # ============================================================
    # Dans l'ancien code, cette inversion était refaite dans chaque
    # boucle d'assemblage
    invjac = np.linalg.inv(jac)      # shape = (ne, ngp, 3, 3)

    # ============================================================
    # 6) Calcul des gradients physiques des fonctions de forme
    # ============================================================
    # Les gradients gN sont donnés dans l'élément de référence.
    # Pour travailler dans le domaine physique, on applique :
    #
    #     gradN_phys = invJ @ gradN_ref
    #
    # On stocke :
    # gradN_phys[e, g, a, :]
    #   = gradient physique de la fonction de forme locale a
    #     dans l'élément e, au point de Gauss g
    gradN_phys = np.zeros((ne, ngp, nloc, 3), dtype=np.float64)

    for e in range(ne):
        for g in range(ngp):
            for a in range(nloc):
                gradN_phys[e, g, a, :] = invjac[e, g] @ gN[g, a]

    # ============================================================
    # 7) Nombre total de DDL compacts
    # ============================================================
    nn = int(np.max(dof_indices) + 1)

    # ============================================================
    # 8) On renvoie tout dans un dictionnaire
    # ============================================================
    return {
        "ne": ne,
        "ngp": ngp,
        "nloc": nloc,
        "nn": nn,
        "conn": conn,
        "dof_indices": dof_indices,
        "det": det,
        "xphys": xphys,
        "jac": jac,
        "invjac": invjac,
        "w": w,
        "N": N,
        "gN": gN,
        "gradN_phys": gradN_phys,
    }


def assemble_residual(U, U_old, M, dt,
                      newton_data,
                      kappa_fun, K_cap,
                      r_growth,
                      dirichlet_dofs=None, dirichlet_vals=None):
    """
    Assemble le résidu global R(U) du problème non linéaire.

    On cherche à résoudre à chaque pas de temps :
        R(U) = 0

    avec ici :
        R(U) = (M/dt)(U - U_old) + R_diff(U) - R_reac(U)

    où :
    - le premier terme correspond au terme temporel implicite,
    - R_diff(U) vient du terme de diffusion non linéaire,
    - R_reac(U) vient du terme de réaction logistique.

    Paramètres
    ----------
    U : ndarray
        Approximation courante de la solution à t^{n+1}.
    U_old : ndarray
        Solution connue au temps précédent t^n.
    M : sparse matrix
        Matrice de masse.
    dt : float
        Pas de temps.
    newton_data : dict
        Données FEM pré-calculées par preprocess_newton_data().
    kappa_fun : callable
        Fonction diffusivité non linéaire kappa(u, x).
    K_cap : callable
        Capacité de charge locale K(x).
    r_growth : float
        Taux de croissance logistique.
    dirichlet_dofs : array-like, optionnel
        Indices des DDL soumis à Dirichlet.
    dirichlet_vals : array-like, optionnel
        Valeurs imposées sur ces DDL.

    Retour
    ------
    R : ndarray
        Vecteur résidu global.
    """

    # ============================================================
    # 1) Lecture des données pré-calculées
    # ============================================================
    ne = newton_data["ne"]
    ngp = newton_data["ngp"]
    nloc = newton_data["nloc"]
    nn = newton_data["nn"]

    dof_indices = newton_data["dof_indices"]
    det = newton_data["det"]
    xphys = newton_data["xphys"]
    w = newton_data["w"]
    N = newton_data["N"]
    gradN_phys = newton_data["gradN_phys"]

    # ============================================================
    # 2) Initialisation des contributions du résidu
    # ============================================================
    R1 = (M @ (U - U_old)) / dt
    R2 = np.zeros(nn)
    R3 = np.zeros(nn)

    # ============================================================
    # 3) Boucle sur les éléments
    # ============================================================
    for e in range(ne):
        # Indices globaux compacts des DDL de l'élément e
        idx = dof_indices[e]

        # Valeurs nodales de U restreintes à l'élément e
        Ue = U[idx]

        # ========================================================
        # 4) Boucle sur les points de Gauss de l'élément
        # ========================================================
        for g in range(ngp):
            xg = xphys[e, g]   # coordonnées physiques du point de Gauss
            wg = w[g]          # poids de quadrature
            detg = det[e, g]   # |det(J)| au point de Gauss

            # ----------------------------------------------------
            # Reconstruction de u_g et grad(u)_g au point de Gauss
            # ----------------------------------------------------
            u_g = 0.0
            grad_u_g = np.zeros(3)

            for b in range(nloc):
                u_g += Ue[b] * N[g, b]
                grad_u_g += Ue[b] * gradN_phys[e, g, b]

            # ----------------------------------------------------
            # Évaluation des coefficients non linéaires au point g
            # ----------------------------------------------------
            kappa_g = float(kappa_fun(u_g, xg))
            K_g = float(K_cap(xg))

            # ----------------------------------------------------
            # Ajout des contributions au résidu local/global
            # ----------------------------------------------------
            for a in range(nloc):
                Ia = int(idx[a])                  # indice global du DDL local a
                Na = N[g, a]                     # fonction de forme locale
                gradNa = gradN_phys[e, g, a]     # gradient physique associé

                # Terme de diffusion :
                # ∫ kappa(u,x) grad(u) · grad(Na)
                R2[Ia] += wg * detg * kappa_g * np.dot(grad_u_g, gradNa)

                # Terme de réaction :
                # ∫ r u (1 - u/K) Na
                R3[Ia] += wg * detg * r_growth * u_g * (1.0 - u_g / K_g) * Na

    # ============================================================
    # 5) Résidu total
    # ============================================================
    R = R1 + R2 - R3

    # ============================================================
    # 6) Imposition des conditions de Dirichlet
    # ============================================================
    if dirichlet_dofs is not None and dirichlet_vals is not None:
        R[dirichlet_dofs] = U[dirichlet_dofs] - dirichlet_vals

    return R


def assemble_jacobian(U, M, dt,
                      newton_data,
                      kappa_fun, dkappa_du, K_cap,
                      r_growth,
                      dirichlet_dofs=None):
    """
    Assemble la jacobienne J(U) du résidu.

    Mathématiquement :
        J(U) = dR/dU

    Elle contient trois contributions :
    - J1 : dérivée du terme temporel
    - J2 : dérivée du terme de diffusion non linéaire
    - J3 : dérivée du terme de réaction logistique

    Paramètres
    ----------
    U : ndarray
        Approximation courante de la solution.
    M : sparse matrix
        Matrice de masse.
    dt : float
        Pas de temps.
    newton_data : dict
        Données FEM pré-calculées.
    kappa_fun : callable
        Diffusivité non linéaire kappa(u, x).
    dkappa_du : callable
        Dérivée de la diffusivité par rapport à u.
    K_cap : callable
        Capacité de charge locale K(x).
    r_growth : float
        Taux de croissance logistique.
    dirichlet_dofs : array-like, optionnel
        DDL soumis à une condition de Dirichlet.

    Retour
    ------
    J : sparse CSR matrix
        Jacobienne globale du résidu.
    """

    # ============================================================
    # 1) Lecture des données pré-calculées
    # ============================================================
    ne = newton_data["ne"]
    ngp = newton_data["ngp"]
    nloc = newton_data["nloc"]
    nn = newton_data["nn"]

    dof_indices = newton_data["dof_indices"]
    det = newton_data["det"]
    xphys = newton_data["xphys"]
    w = newton_data["w"]
    N = newton_data["N"]
    gradN_phys = newton_data["gradN_phys"]

    # ============================================================
    # 2) Initialisation des trois blocs de la jacobienne
    # ============================================================
    J1 = (M / dt).tolil()
    J2 = lil_matrix((nn, nn), dtype=np.float64)
    J3 = lil_matrix((nn, nn), dtype=np.float64)

    # ============================================================
    # 3) Boucle sur les éléments
    # ============================================================
    for e in range(ne):
        idx = dof_indices[e]
        Ue = U[idx]

        # ========================================================
        # 4) Boucle sur les points de Gauss
        # ========================================================
        for g in range(ngp):
            xg = xphys[e, g]
            wg = w[g]
            detg = det[e, g]

            # ----------------------------------------------------
            # Reconstruction de u_g et grad(u)_g
            # ----------------------------------------------------
            u_g = 0.0
            grad_u_g = np.zeros(3)

            for b in range(nloc):
                u_g += Ue[b] * N[g, b]
                grad_u_g += Ue[b] * gradN_phys[e, g, b]

            # ----------------------------------------------------
            # Évaluation des coefficients non linéaires
            # ----------------------------------------------------
            K_g = float(K_cap(xg))
            kappa_g = float(kappa_fun(u_g, xg))
            dkappa_g = float(dkappa_du(u_g, xg))

            # Dérivée de f(u) = r u (1 - u/K)
            # f'(u) = r (1 - 2u/K)
            df_du = r_growth * (1.0 - 2.0 * u_g / K_g)

            # ----------------------------------------------------
            # Boucles locales
            # ----------------------------------------------------
            for b in range(nloc):
                Nb = N[g, b]
                Ib = int(idx[b])
                gradNb = gradN_phys[e, g, b]

                for a in range(nloc):
                    Na = N[g, a]
                    Ia = int(idx[a])
                    gradNa = gradN_phys[e, g, a]

                    # Partie "linéaire" de la diffusion :
                    # ∫ kappa(u,x) grad(Nb)·grad(Na)
                    term_lin = kappa_g * np.dot(gradNb, gradNa)

                    # Partie due à la dépendance de kappa en u :
                    # ∫ dkappa/du * Nb * (grad(u)·grad(Na))
                    term_nonlin = dkappa_g * Nb * np.dot(grad_u_g, gradNa)

                    J2[Ia, Ib] += (term_lin + term_nonlin) * wg * detg

                    # Dérivée du terme de réaction :
                    # ∫ f'(u) Nb Na
                    J3[Ia, Ib] += df_du * Nb * Na * wg * detg

    # ============================================================
    # 5) Jacobienne totale
    # ============================================================
    J = J1 + J2 - J3
    J = J.tolil()

    # ============================================================
    # 6) Imposition des conditions de Dirichlet dans la matrice
    # ============================================================
    # Pour un DDL de Dirichlet i :
    # - on remplace la ligne i par [0 ... 1 ... 0]
    # - on annule la colonne i dans les autres lignes
    #
    if dirichlet_dofs is not None:
        for i in dirichlet_dofs:
            J.rows[i] = [i]
            J.data[i] = [1.0]

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
    newton_data,
    kappa_fun, dkappa_du, K_cap, r_growth,
    dirichlet_dofs=None, dirichlet_vals=None,
    tol=1e-5, max_iter=20):
    """
    Résout le système non linéaire R(U)=0 par la méthode de Newton-Raphson.

    À chaque itération k :
        1. on assemble le résidu R(U^k)
        2. on assemble la jacobienne J(U^k)
        3. on résout :
               J(U^k) deltaU = -R(U^k)
        4. on met à jour :
               U^{k+1} = U^k + deltaU

    Paramètres
    ----------
    U_init : ndarray
        Guess initial pour Newton, en général U_old.
    U_old : ndarray
        Solution au pas de temps précédent.
    M : sparse matrix
        Matrice de masse.
    dt : float
        Pas de temps.
    newton_data : dict
        Données FEM pré-calculées.
    kappa_fun : callable
        Diffusivité non linéaire.
    dkappa_du : callable
        Dérivée de la diffusivité.
    K_cap : callable
        Capacité de charge locale.
    r_growth : float
        Taux de croissance logistique.
    dirichlet_dofs : array-like, optionnel
        DDL imposés.
    dirichlet_vals : array-like, optionnel
        Valeurs imposées sur ces DDL.
    tol : float
        Tolérance d'arrêt.
    max_iter : int
        Nombre maximum d'itérations de Newton.

    Retour
    ------
    U : ndarray
        Solution convergée au pas de temps courant.
    """

    # ============================================================
    # 1) Initialisation
    # ============================================================
    U = U_init.copy()

    # ============================================================
    # 2) Boucle de Newton
    # ============================================================
    for k in range(max_iter):

        # --------------------------------------------------------
        # Étape A : assemblage du résidu
        # --------------------------------------------------------
        R = assemble_residual(
            U, U_old, M, dt,
            newton_data,
            kappa_fun, K_cap, r_growth,
            dirichlet_dofs, dirichlet_vals
        )

        norm_R = np.linalg.norm(R)

        # Si le résidu est suffisamment petit, on considère
        # que Newton a convergé.
        if norm_R < tol:
            print(f"Newton convergé en {k} itérations | ||R|| = {norm_R:.3e}")
            break

        # --------------------------------------------------------
        # Étape B : assemblage de la jacobienne
        # --------------------------------------------------------
        J = assemble_jacobian(
            U, M, dt,
            newton_data,
            kappa_fun, dkappa_du, K_cap, r_growth,
            dirichlet_dofs
        )

        # --------------------------------------------------------
        # Étape C : résolution du système linéaire
        # --------------------------------------------------------
        deltaU = spsolve(J, -R)

        # --------------------------------------------------------
        # Étape D : mise à jour de la solution
        # --------------------------------------------------------
        U += deltaU

        # Critère secondaire : si la correction devient très petite,
        # on peut aussi arrêter Newton.
        if np.linalg.norm(deltaU) < tol:
            print(f"Newton arrêté par petite correction en {k+1} itérations | ||deltaU|| = {np.linalg.norm(deltaU):.3e}")
            break

    return U