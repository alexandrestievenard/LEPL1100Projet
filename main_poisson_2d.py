# main_poisson_1d.py
import argparse
import numpy as np
import matplotlib.pyplot as plt
import gmsh

from gmsh_utils import (
    gmsh_init, gmsh_finalize, build_1d_mesh, build_2d_mesh, format_2d_mesh,
    prepare_quadrature_and_basis, get_jacobians, end_dofs_from_nodes
)
from stiffness import assemble_stiffness_and_rhs
from dirichlet import solve_dirichlet
from errors import compute_L2_H1_errors

from plot_utils import plot_fe_solution_high_order

def main(geo_filename, L, order):

    gmsh_init("poisson_2d")
    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags = build_2d_mesh(geo_filename, L, order)
    coords_nodes, elements, elements_idx = format_2d_mesh(elemType, nodeTags, nodeCoords, elemTags, elemNodeTags)

    xi, w, N, gN = prepare_quadrature_and_basis(elemType, order)
    jac, det, coords_gp = get_jacobians(elemType, xi)

    kappa = lambda x: 1.0
    f = lambda x: 1.0

    K_lil, F = assemble_stiffness_and_rhs(
    coords_nodes.shape[0],
    elemTags,
    elements_idx,
    jac,
    det,
    coords_gp,
    w,
    N,
    gN,
    kappa,
    f
)
    K = K_lil.tocsr()

    U = solve_dirichlet(K, F, dir_dofs, dir_vals)

    # order+7 is a bit of a hack to get a very accurate quadrature for the error computation
    errL2, errH1s, errH1 =  computeErrors1d(elemType, elemTags, elemNodeTags, U, order+7, u_exact, grad_exact)

    fig, ax = plt.subplots()
    plot_fe_solution_high_order(elemType, elemNodeTags, nodeCoords, U, M=200, show_nodes=True, ax=ax, label="FE solution")
    gmsh_finalize()

    x_plot = np.linspace(0.0, L, 200)
    u_plot = u_exact([x_plot])
    ax.plot(x_plot, u_plot, "k--", label="Exact solution")
    ax.legend()
    plt.show()

    return errL2, errH1s, errH1

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Poisson 1D with Gmsh high-order FE")
    parser.add_argument("-order", type=int, default=1)
    parser.add_argument("-cl1", type=float, default=0.1)
    parser.add_argument("-cl2", type=float, default=0.1)
    parser.add_argument("-L", type=float, default=1.0)
    args = parser.parse_args()
    order = args.order
    L = args.L
    cl1 = args.cl1
    cl2 = args.cl2

    #errL2, errH1s, errH1 = main(cl1, cl2, L, order)

    # Visualize 2D grid
    gmsh.initialize()

    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags = build_2d_mesh(
        "square.geo",
        mesh_size=1,
        order=1
    )
    
    coords, elements, elements_idx = format_2d_mesh(elemType, nodeTags, nodeCoords, elemTags, elemNodeTags)
    print("\n----------------elements--------------------\n")
    print(elements)
    print("\n----------------elements_idx--------------------\n")
    print(elements_idx)