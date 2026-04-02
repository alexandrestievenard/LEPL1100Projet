# main_diffusion_1d.py
import argparse
import numpy as np

from gmsh_utils import (
    getPhysical, gmsh_init, gmsh_finalize, open_2d_mesh,
    prepare_quadrature_and_basis, get_jacobians,
    border_dofs_from_tags
)
from stiffness import assemble_rhs_neumann, assemble_stiffness_and_rhs
from mass import assemble_mass
from dirichlet import theta_step
from plot_utils import setup_interactive_figure, plot_mesh_2d, plot_fe_solution_2d
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Diffusion 1D with theta-scheme (Gmsh high-order FE)")
    parser.add_argument("-order", type=int, default=1)

    parser.add_argument("--theta", type=float, default=1.0, help="1: implicit Euler, 0.5: Crank-Nicolson, 0: explicit")
    parser.add_argument("--dt", type=float, default=1.0e-03)
    parser.add_argument("--nsteps", type=int, default=500)
    args = parser.parse_args()

    gmsh_init("panpan_2d")

    dt = args.dt
    nstep = args.nsteps
    T = dt * nstep

    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags = open_2d_mesh(
        msh_filename="panpan.msh", order=args.order
    )

    plot_mesh_2d(elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags)

    unique_dofs_tags = np.unique(elemNodeTags)

    num_dofs = len(unique_dofs_tags)
    max_tag = int(np.max(nodeTags))
    dof_coords = np.zeros((num_dofs, 3))
    all_coords = nodeCoords.reshape(-1, 3)
    tag_to_dof = np.full(max_tag + 1, -1, dtype=int)
    for i, tag in enumerate(unique_dofs_tags):
        tag_to_dof[int(tag)] = i
        dof_coords[i] = all_coords[i]
    # Note: We create a mapping from Gmsh node tags to our dof indices, and we also store the coordinates of the dofs. This will be useful for assembling the system and for plotting.

    xi, w, N, gN = prepare_quadrature_and_basis(elemType, args.order)
    jac, det, coords = get_jacobians(elemType, xi)

    elemTypeNeuBnd, elemTagsNeuBnd, elemNodeTagsNeuBnd, entityTag = getPhysical("InnerBoundary")

    xiBnd, wBnd, NBnd, gNBnd = prepare_quadrature_and_basis(elemTypeNeuBnd, args.order)
    jacBnd, detBnd, coordsBnd = get_jacobians(elemTypeNeuBnd, xiBnd, tag=entityTag)
    # Note: For the Neumann part, we integrate on the boundary, so we use the boundary elements and their corresponding quadrature and jacobians.

    def kappa(x): return 1.0
    def f_source(x, t): return 0.0
    def u0(x): return 0.0
    def u_outer(x, t): return 1.0*np.sin(11*np.pi * t/T)
    def g_inner(x, t): return 1.0*np.sin(31*np.pi * t/T)

    M_lil = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)
    K_lil, F0 = assemble_stiffness_and_rhs(elemTags, elemNodeTags, jac, det, coords, w, N, gN, kappa, lambda x: f_source(x, 0), tag_to_dof)
    # Note: We compute the stifness, mass and rhs part depending on the source term only once, since the source and kappa are time-independent. If they were time-dependent, we would need to re-assemble at each time step (or use a more efficient approach for the time-dependence).

    M = M_lil.tocsr()
    K = K_lil.tocsr()

    U = np.array([u0(x) for x in dof_coords], dtype=float)

    outer_dofs = border_dofs_from_tags(bnds_tags[0], tag_to_dof)
    dir_dofs = outer_dofs

    _, ax = setup_interactive_figure()

    for step in range(args.nsteps):
        t = step * dt

        Fn = assemble_rhs_neumann(F0, elemTagsNeuBnd, elemNodeTagsNeuBnd, jacBnd, detBnd, coordsBnd, wBnd, NBnd, gNBnd, lambda x: g_inner(x, t), tag_to_dof)

        Fnp1 = assemble_rhs_neumann(F0, elemTagsNeuBnd, elemNodeTagsNeuBnd, jacBnd, detBnd, coordsBnd, wBnd, NBnd, gNBnd, lambda x: g_inner(x, t+dt), tag_to_dof)
        # Note: We re-assemble the Neumann part of the rhs at each time step, since the Neumann data is time-dependent. If it were time-independent, we could compute it once and reuse it. We do it at time n and n+1 since the theta-scheme requires both.

        dir_vals_np1 = u_outer(np.zeros(3), t+dt)*np.ones_like(outer_dofs)
        # The Dirichlet values are space independent in this example, so we can just evaluate them at the origin. If they were space-dependent, we would need to evaluate them at the coordinates of the Dirichlet dofs.

        U = theta_step(M, K, Fn, Fnp1, U, dt=dt, theta=args.theta, dirichlet_dofs=dir_dofs, dir_vals_np1=dir_vals_np1)

        ax.clear()

        plot_fe_solution_2d(
            elemNodeTags=elemNodeTags,
            nodeTags=nodeTags,
            nodeCoords=nodeCoords,
            U=U,
            tag_to_dof=tag_to_dof,
            show_mesh=False,
            ax=ax
        )

        ax.set_title(f"t = {step * args.dt:.4f}   (theta={args.theta})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axis('equal')

        plt.pause(0.01)

    gmsh_finalize()

if __name__ == "__main__":
    main()
