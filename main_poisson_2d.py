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
    return x[0]**2 + x[1]**2

def grad_exact(x):
    return np.array([2*x[0], 2*x[1], 0.0])

def rhs_fun(x):
    return -4.0


def kappa_fun(x):
    return 1.0


def convergence_rates(hs, errs):
    rates = [np.nan]
    for i in range(1, len(errs)):
        r = np.log(errs[i-1] / errs[i]) / np.log(hs[i-1] / hs[i])
        rates.append(r)
    return rates


def main(geo_filename, mesh_size, order, L):
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

    gmsh_finalize()

    plot_solution_2d(coords_nodes, elements_idx, U)

    err = np.max(np.abs(U - np.array([u_exact(x) for x in coords_nodes])))
    print("max error =", err)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Poisson FE verification")
    parser.add_argument("-order", type=int, default=1)
    parser.add_argument("-L", type=float, default=1.0)
    args = parser.parse_args()

    order = args.order
    L = args.L

    main(
        "square.geo",
        mesh_size=0.05,
        order=1,
        L=1.0
    )
