
import time
import gmsh
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

from gmsh_utils import (
    gmsh_init, gmsh_finalize, open_2d_mesh,
    prepare_quadrature_and_basis, get_jacobians,
    border_dofs_from_tags
)
from mass import assemble_mass
from stiffness_non_linear import assemble_stiffness_and_rhs
from dirichlet import theta_step


L = 10.0   # côté du carré [km]


def u_exact(xy, t):
    """Solution exacte imposée pour le test de convergence."""
    x, y = xy[0], xy[1]
    a = np.pi / L
    return np.sin(a * x) * np.sin(a * y) * np.exp(-t)


def grad_exact(xy, t):
    """Gradient analytique de la solution exacte."""
    x, y = xy[0], xy[1]
    a = np.pi / L
    val = np.exp(-t)

    du_dx = a * np.cos(a * x) * np.sin(a * y) * val
    du_dy = a * np.sin(a * x) * np.cos(a * y) * val

    return np.array([du_dx, du_dy, 0.0])


def source_exact(xy, t):
    """
    Terme source manufacturé.

    On part de :
        u_t - div(kappa(u) grad u) = r u (1 - u/K) + S

    donc :
        S = u_t - div(kappa(u) grad u) - r u (1 - u/K)
    """
    x, y = xy[0], xy[1]
    a = np.pi / L

    u = u_exact(xy, t)
    grad_u = grad_exact(xy, t)

    k0 = 5.0
    alpha = 0.02

    kappa = k0 / (1.0 + alpha * u)
    dkappa_du = -alpha * k0 / (1.0 + alpha * u)**2

    laplacien = -2.0 * a**2 * u
    grad_u_norm2 = grad_u[0]**2 + grad_u[1]**2

    diff_term = kappa * laplacien + dkappa_du * grad_u_norm2

    du_dt = -u

    r = 1.0
    K = 50.0
    reaction = r * u * (1.0 - u / K)

    return du_dt - diff_term - reaction


def generate_square_mesh(Lx=10.0, Ly=10.0, h=0.5, order=1, filename="square.msh"):
    """
    Génère un maillage triangulaire du carré [0,Lx] x [0,Ly].

    Le bord extérieur est placé dans le groupe physique "OuterBoundary",
    ce qui permet d'imposer ensuite une condition de Dirichlet homogène.
    """
    gmsh.initialize()
    gmsh.model.add("square")

    p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, h)
    p1 = gmsh.model.geo.addPoint(Lx, 0.0, 0.0, h)
    p2 = gmsh.model.geo.addPoint(Lx, Ly, 0.0, h)
    p3 = gmsh.model.geo.addPoint(0.0, Ly, 0.0, h)

    l0 = gmsh.model.geo.addLine(p0, p1)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p0)

    cl = gmsh.model.geo.addCurveLoop([l0, l1, l2, l3])
    s = gmsh.model.geo.addPlaneSurface([cl])

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, [l0, l1, l2, l3], tag=10)
    gmsh.model.setPhysicalName(1, 10, "OuterBoundary")

    gmsh.model.addPhysicalGroup(2, [s], tag=1)
    gmsh.model.setPhysicalName(2, 1, "Domain")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)
    gmsh.write(filename)

    gmsh.finalize()
    return filename


def build_test_problem(msh_file, order=1, K_const=50.0):
    """
    Prépare les données FEM nécessaires au test de convergence.

    La géométrie est volontairement simple afin d'isoler l'erreur numérique.
    """
    gmsh_init()

    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags = \
        open_2d_mesh(msh_file, order)

    # mapping tags Gmsh -> indices DDL compacts
    unique_dofs_tags = np.unique(elemNodeTags)
    num_dofs = len(unique_dofs_tags)

    max_tag = int(np.max(nodeTags))
    tag_to_dof = np.full(max_tag + 1, -1, dtype=int)

    all_coords = nodeCoords.reshape(-1, 3)
    tag_to_node_index = {int(tag): i for i, tag in enumerate(nodeTags)}

    dof_coords = np.zeros((num_dofs, 3))

    for i, tag in enumerate(unique_dofs_tags):
        tag_int = int(tag)
        tag_to_dof[tag_int] = i
        dof_coords[i] = all_coords[tag_to_node_index[tag_int]]

    #Quadrature et géométrie
    xi, w, N, gN = prepare_quadrature_and_basis(elemType, order)
    jac, det, coords = get_jacobians(elemType, xi)

    #Condition de Dirichlet homogène sur tout le bord
    bnd_names = [name for name, _ in bnds]
    outer_tags = bnds_tags[bnd_names.index("OuterBoundary")]

    dir_dofs = border_dofs_from_tags(outer_tags, tag_to_dof).astype(int)
    dir_vals = np.zeros(len(dir_dofs), dtype=float)

    #Capacité de charge constante
    K_nodal = K_const * np.ones(num_dofs, dtype=float)

    # Matrice de masse 
    M_lil = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)
    M = M_lil.tocsr()
    M_lump = np.array(M.sum(axis=1)).flatten()

    # Prétraitement Newton
    from newton_solver import preprocess_newton_data

    newton_data = preprocess_newton_data(
        elemTags=elemTags,
        conn=elemNodeTags,
        jac=jac,
        det=det,
        xphys=coords,
        w=w,
        N=N,
        gN=gN,
        tag_to_dof=tag_to_dof,
        K_nodal=K_nodal
    )

    # Condition initiale exacte 
    U0 = np.array([
        u_exact(dof_coords[i, :2], 0.0)
        for i in range(num_dofs)
    ], dtype=float)

    def kappa_fun(u, x):
        return 5.0 / (1.0 + 0.02 * u)

    def dkappa_du(u, x):
        return -0.02 * 5.0 / (1.0 + 0.02 * u)**2

    return {
        # maillage
        "elemType": elemType,
        "nodeTags": nodeTags,
        "nodeCoords": nodeCoords,
        "elemTags": elemTags,
        "elemNodeTags": elemNodeTags,
        "bnds": bnds,
        "bnds_tags": bnds_tags,

        # DDLs
        "num_dofs": num_dofs,
        "tag_to_dof": tag_to_dof,
        "dof_coords": dof_coords,

        # quadrature
        "xi": xi,
        "w": w,
        "N": N,
        "gN": gN,
        "jac": jac,
        "det": det,
        "coords": coords,

        # matrices
        "M": M,
        "M_lump": M_lump,

        # conditions aux limites
        "dir_dofs": dir_dofs,
        "dir_vals": dir_vals,

        # champs physiques
        "K_nodal": K_nodal,
        "kappa_fun": kappa_fun,
        "dkappa_du": dkappa_du,
        "R_GROWTH": 1.0,

        # Newton et condition initiale
        "newton_data": newton_data,
        "U0": U0,
    }

def imex_step_with_source(U_old, problem, dt, theta, t):
    """
    Effectue un pas IMEX en ajoutant le terme source manufacturé.

    La diffusion est traitée implicitement avec kappa évalué en u^n.
    La réaction reste explicite et lumpée, comme dans imex_solver.py.
    """
    M = problem["M"]
    M_lump = problem["M_lump"]

    dir_dofs = problem["dir_dofs"]
    dir_vals = problem["dir_vals"]

    # Pour theta = 1, la source est évaluée au temps futur t^{n+1}.
    K_lil, F_source = assemble_stiffness_and_rhs(
        problem["elemTags"],
        problem["elemNodeTags"],
        problem["jac"],
        problem["det"],
        problem["coords"],
        problem["w"],
        problem["N"],
        problem["gN"],
        U_old,
        problem["kappa_fun"],
        lambda x: source_exact(x, t + dt),
        problem["tag_to_dof"]
    )
    K_mat = K_lil.tocsr()

    U_pos = np.maximum(U_old, 0.0)
    f_react = problem["R_GROWTH"] * U_pos * (1.0 - U_pos / problem["K_nodal"])

    F_total = F_source + f_react * M_lump

    U_new = theta_step(
        M, K_mat,
        F_total, F_total,
        U_old,
        dt=dt,
        theta=theta,
        dirichlet_dofs=dir_dofs,
        dir_vals_np1=dir_vals
    )

    U_new[dir_dofs] = dir_vals
    return U_new


def newton_solver_with_source(U_init, U_old, problem, dt, t, tol=1e-8, max_iter=20):
    """
    Résout le pas implicite non linéaire avec Newton-Raphson.

    Le résidu et la jacobienne proviennent de newton_solver.py.
    On ajoute uniquement le terme source manufacturé dans le résidu.
    """
    from newton_solver import assemble_residual as assemble_residual_orig
    from newton_solver import assemble_jacobian as assemble_jacobian_orig

    M = problem["M"]
    newton_data = problem["newton_data"]

    dir_dofs = problem["dir_dofs"]
    dir_vals = problem["dir_vals"]

    kappa_fun = problem["kappa_fun"]
    dkappa_du = problem["dkappa_du"]
    r_growth = problem["R_GROWTH"]

    # Assemblage du vecteur source au temps t^{n+1}
    nn = problem["num_dofs"]
    F_source = np.zeros(nn)

    ne = len(problem["elemTags"])
    ngp = len(problem["w"])
    nloc = int(len(problem["elemNodeTags"]) // ne)

    conn = np.asarray(problem["elemNodeTags"]).reshape(ne, nloc)
    det = problem["det"].reshape(ne, ngp)
    coords = problem["coords"].reshape(ne, ngp, 3)

    w = problem["w"]
    N = problem["N"].reshape(ngp, nloc)

    dof_indices = problem["tag_to_dof"][conn]

    for e in range(ne):
        idx = dof_indices[e]

        for g in range(ngp):
            xg = coords[e, g]
            wg = w[g]
            detg = det[e, g]
            Sg = source_exact(xg, t)

            for a in range(nloc):
                Ia = int(idx[a])
                F_source[Ia] += wg * Sg * N[g, a] * detg

    # Les DDLs imposés sont traités séparément dans le résidu.
    F_source[dir_dofs] = 0.0

    def assemble_residual_with_source(U):
        """
        Résidu du problème manufacturé :
            R_total = R_physique - F_source
        """
        R = assemble_residual_orig(
            U, U_old, M, dt,
            newton_data, kappa_fun, r_growth,
            dirichlet_dofs=None,
            dirichlet_vals=None
        )

        R -= F_source
        R[dir_dofs] = U[dir_dofs] - dir_vals

        return R

    U = U_init.copy()

    for _ in range(max_iter):
        R = assemble_residual_with_source(U)

        if np.linalg.norm(R) < tol:
            break

        J = assemble_jacobian_orig(
            U, M, dt,
            newton_data, kappa_fun, dkappa_du, r_growth,
            dir_dofs
        )

        deltaU = spsolve(J, -R)
        U += deltaU

        if np.linalg.norm(deltaU) < tol:
            break

    U[dir_dofs] = dir_vals
    return U


def run_test_simulation(problem, dt, nsteps, method, theta=1.0, save_every=5, print_every=5):
    """
    Lance une simulation complète jusqu'à T_final.

    Retourne la solution finale et le temps CPU associé.
    """
    U = problem["U0"].copy()
    dir_dofs = problem["dir_dofs"]
    dir_vals = problem["dir_vals"]

    t_start = time.perf_counter()

    # Stockage de l'erreur avec le temps
    times = []
    errors_L2 = []
    errors_H1 = []

    for step in range(nsteps):
        t = step * dt

        if method == "imex":
            U = imex_step_with_source(U, problem, dt, theta, t)

        elif method == "newton":
            U = newton_solver_with_source(
                U_init=U.copy(),
                U_old=U,
                problem=problem,
                dt=dt,
                t=t + dt,
                tol=1e-10,
                max_iter=20
            )

        else:
            raise ValueError(f"Méthode inconnue : {method}")

        U[dir_dofs] = dir_vals

        if step % save_every == 0:
            err_L2, _, err_H1 = compute_error(problem, U, t + dt)
            times.append(t + dt)
            errors_L2.append(err_L2)
            errors_H1.append(err_H1)

        if step % print_every == 0 or step == nsteps - 1:
            print(
                f"  t={t+dt:.3f} | max(U)={np.max(U):.4f} "
                f"| mean(U)={np.mean(U):.4f}"
            )

    cpu_time = time.perf_counter() - t_start
    return U, cpu_time


def compute_error(problem, U_final, t_final):
    """
    Calcule les erreurs L2, H1_semi et H1 à l'instant final.

    On utilise directement les données FEM déjà préparées dans problem.
    """
    elemNodeTags = problem["elemNodeTags"]
    tag_to_dof = problem["tag_to_dof"]

    w = problem["w"]
    N = problem["N"]
    gN = problem["gN"]
    jac = problem["jac"]
    det = problem["det"]
    coords = problem["coords"]

    ne = len(problem["elemTags"])
    ngp = len(w)
    nloc = int(len(elemNodeTags) // ne)

    det_arr = det.reshape(ne, ngp)
    jac_arr = jac.reshape(ne, ngp, 3, 3)
    coords_arr = coords.reshape(ne, ngp, 3)

    conn_tags = np.asarray(elemNodeTags).reshape(ne, nloc)
    conn = tag_to_dof[conn_tags]

    N_arr = N.reshape(ngp, nloc)
    gN_arr = gN.reshape(ngp, nloc, 3)

    I_L2 = 0.0
    I_H1 = 0.0

    for e in range(ne):
        nodes = conn[e]
        Ue = U_final[nodes]

        for g in range(ngp):
            xg = coords_arr[e, g]
            wg = w[g]
            detg = det_arr[e, g]

            invjac = np.linalg.inv(jac_arr[e, g])

            uh = 0.0
            grad_uh = np.zeros(3)

            for a in range(nloc):
                Na = N_arr[g, a]
                gradNa = invjac @ gN_arr[g, a]

                uh += Na * Ue[a]
                grad_uh += gradNa * Ue[a]

            uex = u_exact(xg, t_final)
            gex = grad_exact(xg, t_final)

            du = uh - uex
            dg = grad_uh - gex

            I_L2 += wg * du * du * detg
            I_H1 += wg * np.dot(dg, dg) * detg

    err_L2 = np.sqrt(max(I_L2, 0.0))
    err_H1_semi = np.sqrt(max(I_H1, 0.0))
    err_H1 = np.sqrt(err_L2**2 + err_H1_semi**2)

    return err_L2, err_H1_semi, err_H1

def main():
    order = 1
    mesh_size = 0.25
    msh_file = "square.msh"

    generate_square_mesh(
        Lx=L,
        Ly=L,
        h=mesh_size,
        order=order,
        filename=msh_file
    )

    problem = build_test_problem(msh_file, order=order, K_const=50.0)
    print(f"Maillage généré : {problem['num_dofs']} nœuds.")

    # -------------------------------------------------------------------------
    # 1. Étude de l'erreur en fonction du temps
    # -------------------------------------------------------------------------
    dt_single = 0.1
    T_final = 1.0
    nsteps_single = int(T_final / dt_single)

    print(f"\n=== Évolution temporelle de l'erreur (dt = {dt_single}) ===")

    # IMEX
    _, _, t_imex, errL2_imex, _ = run_test_simulation(
        problem, dt_single, nsteps_single,
        method="imex", theta=1.0,
        save_every=1, print_every=nsteps_single//4
    )

    # Newton
    _, _, t_newton, errL2_newton, _ = run_test_simulation(
        problem, dt_single, nsteps_single,
        method="newton",
        save_every=1, print_every=nsteps_single//4
    )

    plt.figure()
    plt.plot(t_imex, errL2_imex, 'o-', label='IMEX (θ=1)')
    plt.plot(t_newton, errL2_newton, 's-', label='Newton')
    plt.xlabel('Temps (années)')
    plt.ylabel('Erreur L2')
    plt.title(f"Évolution de l'erreur - dt = {dt_single}")
    plt.legend()
    plt.grid(True)
    plt.savefig('error_vs_time.png')
    plt.show()

    # -------------------------------------------------------------------------
    # 2. Étude de convergence (erreur finale)
    # -------------------------------------------------------------------------
    dt_list = [0.05, 0.025, 0.0125, 0.00625]
    results = []

    for dt in dt_list:
        nsteps = int(T_final / dt)

        print(f"\n--- dt = {dt} (nsteps={nsteps}) ---")

        U_imex, cpu_imex, _, _, _ = run_test_simulation(
            problem, dt, nsteps,
            method="imex",
            theta=1.0,
            save_every=5,
            print_every=max(1, nsteps//5)
        )
        err_imex = compute_error(problem, U_imex, T_final)

        print(
            f"IMEX : CPU = {cpu_imex:.3f} s, "
            f"err_L2 = {err_imex[0]:.2e}, err_H1 = {err_imex[2]:.2e}"
        )

        U_newton, cpu_newton, _, _, _ = run_test_simulation(
            problem, dt, nsteps,
            method="newton",
            save_every=5,
            print_every=max(1, nsteps//5)
        )
        err_newton = compute_error(problem, U_newton, T_final)

        print(
            f"Newton: CPU = {cpu_newton:.3f} s, "
            f"err_L2 = {err_newton[0]:.2e}, err_H1 = {err_newton[2]:.2e}"
        )

        results.append((dt, err_imex[0], cpu_imex, err_newton[0], cpu_newton))

    print("\n=== Récapitulatif ===")
    print("dt      | err_imex (L2) | t_imex (s) | err_newton (L2) | t_newton (s)")

    for dt, e_im, t_im, e_nw, t_nw in results:
        print(f"{dt:6.4f} | {e_im:.2e}      | {t_im:.3f}      | {e_nw:.2e}        | {t_nw:.3f}")

    #Courbes de convergence
    dt_vals = np.array([r[0] for r in results])
    err_im = np.array([r[1] for r in results])
    err_nw = np.array([r[3] for r in results])
    ref = err_im[0] * dt_vals / dt_vals[0]

    plt.figure()
    plt.loglog(dt_vals, err_im, "o-", label="IMEX (θ=1)")
    plt.loglog(dt_vals, err_nw, "s-", label="Newton")
    plt.loglog(dt_vals, ref, "k--", label="ordre 1")
    plt.xlabel("dt (années)")
    plt.ylabel("Erreur L2 à t=1 an")
    plt.legend()
    plt.grid(True)
    plt.savefig("convergence.png")
    plt.show()

    gmsh_finalize()


if __name__ == "__main__":
    main()