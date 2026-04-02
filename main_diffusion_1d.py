# main_diffusion_1d.py
import argparse
import numpy as np

from gmsh_utils import (
    gmsh_init, gmsh_finalize, build_1d_mesh,
    prepare_quadrature_and_basis, get_jacobians, end_dofs_from_nodes
)
from stiffness import assemble_stiffness_and_rhs
from mass import assemble_mass
from dirichlet import theta_step
from plot_utils import plot_fe_solution_high_order, setup_interactive_figure


def main():
    parser = argparse.ArgumentParser(description="Diffusion 1D with theta-scheme (Gmsh high-order FE)")
    parser.add_argument("-order", type=int, default=1)
    parser.add_argument("-cl1", type=float, default=0.05)
    parser.add_argument("-cl2", type=float, default=0.05)
    parser.add_argument("-L", type=float, default=1.0)

    parser.add_argument("--theta", type=float, default=1.0, help="1: implicit Euler, 0.5: Crank-Nicolson, 0: explicit")
    parser.add_argument("--dt", type=float, default=1.0e-04)
    parser.add_argument("--nsteps", type=int, default=500)
    args = parser.parse_args()

    gmsh_init("diffusion_1d")

    L = args.L

    _, elemType, nodeTags, nodeCoords, elemTags, elemNodeTags = build_1d_mesh(
        L=args.L, cl1=args.cl1, cl2=args.cl2, order=args.order
    )

    unique_dofs_tags = np.unique(elemNodeTags)

    num_dofs = len(unique_dofs_tags)
    max_tag = int(np.max(nodeTags))
    dof_coords = np.zeros((num_dofs, 3))
    all_coords = nodeCoords.reshape(-1, 3)
    tag_to_dof = np.full(max_tag + 1, -1, dtype=int)
    for i, tag in enumerate(unique_dofs_tags):
        tag_to_dof[int(tag)] = i
        dof_coords[i] = all_coords[i]

    xi, w, N, gN = prepare_quadrature_and_basis(elemType, args.order)
    jac, det, coords = get_jacobians(elemType, xi)

    def kappa(x): return 1.0
    def f_source(x): return 0.0
    def u0(x): return 1.0*np.sin(np.pi*x/L) + 2.0*np.sin(8*np.pi*x/L)

    K_lil, F = assemble_stiffness_and_rhs(
        elemTags, elemNodeTags, jac, det, coords, w, N, gN, kappa, f_source, tag_to_dof
    )
    M_lil = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)

    K = K_lil.tocsr()
    M = M_lil.tocsr()

    n = len(F)
    U = np.array([u0(x) for x in nodeCoords[::3]], dtype=float)

    left, right = end_dofs_from_nodes(nodeCoords)
    dir_dofs = [left, right]
    dir_vals = np.array([0.0, 0.0], dtype=float)

    fig, ax = setup_interactive_figure(xlim=(0.0, args.L))
    u_min = float(np.min(U))
    u_max = float(np.max(U))
    pad = 0.05 * (u_max - u_min + 1e-14)
    ylim = (u_min - pad, u_max + pad)

    import matplotlib.pyplot as plt

    for step in range(args.nsteps):
        U = theta_step(M, K, F, F, U, dt=args.dt, theta=args.theta, dirichlet_dofs=dir_dofs, dir_vals_np1=dir_vals)

        ax.clear()
        ax.set_xlim(0.0, args.L)
        ax.set_ylim(*ylim)

        plot_fe_solution_high_order(
            elemType=elemType,
            elemNodeTags=elemNodeTags,
            nodeCoords=nodeCoords,
            U=U,
            M=120,
            show_nodes=False,
            ax=ax
        )

        ax.set_title(f"t = {step * args.dt:.4f}   (theta={args.theta})")
        ax.set_xlabel("x")
        ax.set_ylabel(r"$u_h(x,t)$")
        ax.grid(True)

        plt.pause(0.03)

    gmsh_finalize()


if __name__ == "__main__":
    main()
