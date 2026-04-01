# main_poisson_2d.py
import argparse
import numpy as np
import gmsh

from gmsh_utils import (
    gmsh_init, gmsh_finalize, build_2d_mesh, format_2d_mesh,
    prepare_quadrature_and_basis, get_jacobians
)
from stiffness import assemble_stiffness_and_rhs
from dirichlet import solve_dirichlet
from plot_utils import plot_solution_2d
from errors import compute_L2_H1_errors, convergence_rates


def boundary_dofs_square(coords_nodes, L, tol=1e-12):
    x = coords_nodes[:, 0]
    y = coords_nodes[:, 1]

    mask = (
        np.isclose(x, 0.0, atol=tol) |
        np.isclose(x, L,   atol=tol) |
        np.isclose(y, 0.0, atol=tol) |
        np.isclose(y, L,   atol=tol)
    )
    return np.nonzero(mask)[0]


def u_exact(x):
    return np.sin(np.pi*x[0]) * np.sin(np.pi*x[1])

def grad_exact(x):
    return np.array([
        np.pi*np.cos(np.pi*x[0]) * np.sin(np.pi*x[1]),
        np.pi*np.sin(np.pi*x[0]) * np.cos(np.pi*x[1]),
        0.0
    ])

def rhs_fun(x):
    return 2.0 * np.pi**2 * np.sin(np.pi*x[0]) * np.sin(np.pi*x[1])


def kappa_fun(x):
    return 1.0




def main(geo_filename, mesh_size, order, L, do_plot=False):
    gmsh_init("poisson_2d")

    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags = build_2d_mesh(
        geo_filename, mesh_size, order
    )
    coords_nodes, elements, elements_idx = format_2d_mesh(
        nodeTags, nodeCoords, elemTags, elemNodeTags
    )

    xi, w, N, gN = prepare_quadrature_and_basis(elemType, order)
    jac, det, coords_gp = get_jacobians(elemType, xi)

    K_lil, F = assemble_stiffness_and_rhs(
        elemTags,
        elements_idx,
        jac,
        det,
        coords_gp,
        w,
        N,
        gN,
        kappa_fun,
        rhs_fun
    )
    K = K_lil.tocsr()

    dirichlet_dofs = boundary_dofs_square(coords_nodes, L)
    dirichlet_values = np.array(
        [u_exact(coords_nodes[i]) for i in dirichlet_dofs],
        dtype=float
    )

    U = solve_dirichlet(K, F, dirichlet_dofs, dirichlet_values)

    errL2, errH1s, errH1 = compute_L2_H1_errors(
        elemType, elemTags, elements_idx, U,
        xi, w, N, gN, jac, det, coords_gp,
        u_exact, grad_exact
    )

    if do_plot:
        plot_solution_2d(coords_nodes, elements_idx, U)

    gmsh_finalize()

    return errL2, errH1s, errH1, coords_nodes.shape[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Poisson FE verification")
    parser.add_argument("-order", type=int, default=1)
    parser.add_argument("-L", type=float, default=1.0)
    args = parser.parse_args()

    order = args.order
    L = args.L

    mesh_sizes = [0.2, 0.1, 0.05]

    errsL2 = []
    errsH1s = []
    errsH1 = []
    nnodes = []

    for h in mesh_sizes:
        print(f"\n------- mesh_size = {h} ----------")
        errL2, errH1s, errH1, nn = main(
            "square.geo",
            mesh_size=h,
            order=1,
            L=1.0,
            do_plot=True
        )
        errsL2.append(errL2)
        errsH1s.append(errH1s)
        errsH1.append(errH1)
        nnodes.append(nn)

    ratesL2 = convergence_rates(mesh_sizes, errsL2)
    ratesH1s = convergence_rates(mesh_sizes, errsH1s)
    ratesH1 = convergence_rates(mesh_sizes, errsH1)

    print_table = False
    if (print_table):
        print("\nConvergence table:")
        print(f"{'h':>10} {'nnodes':>10} {'errL2':>15} {'rateL2':>10} {'errH1s':>15} {'rateH1s':>10} {'errH1':>15} {'rateH1':>10}")
        for h, nn, eL2, rL2, eH1s, rH1s, eH1, rH1 in zip(
            mesh_sizes, nnodes, errsL2, ratesL2, errsH1s, ratesH1s, errsH1, ratesH1
        ):
            print(f"{h:10.4f} {nn:10d} {eL2:15.6e} {rL2:10.4f} {eH1s:15.6e} {rH1s:10.4f} {eH1:15.6e} {rH1:10.4f}")
