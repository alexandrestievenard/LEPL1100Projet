# plot_utils.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import gmsh


def plot_fe_solution_high_order(
    elemType, elemNodeTags, nodeCoords, U,
    M=80, show_nodes=False, ax=None, label=None
):
    """
    Plot 1D high-order FE solution by sampling each element and evaluating gmsh basis.
    Assumes U is aligned with gmsh's compact node ordering (0..nn-1).
    """
    _, _, _, nloc, _, _ = gmsh.model.mesh.getElementProperties(elemType)

    u = np.linspace(-1.0, 1.0, int(M))
    pts3 = np.zeros((len(u), 3), dtype=float)
    pts3[:, 0] = u
    uvw = pts3.reshape(-1).tolist()

    _, bf, _ = gmsh.model.mesh.getBasisFunctions(elemType, uvw, "Lagrange")
    N = np.asarray(bf, dtype=float).reshape(len(u), nloc)

    if ax is None:
        fig, ax = plt.subplots()

    ne = int(len(elemNodeTags) // nloc)
    _, _, coords_flat = gmsh.model.mesh.getJacobians(elemType, uvw)
    coords = np.asarray(coords_flat, dtype=float).reshape(ne, len(u), 3)

    for e in range(ne):
        tags_e = np.asarray(elemNodeTags[e * nloc:(e + 1) * nloc], dtype=int) - 1
        Ue = U[tags_e]

        x = coords[e, :, 0]
        uh = N @ Ue

        order = np.argsort(x)
        ax.plot(x[order], uh[order], label=label if (e == 0) else None)

    if show_nodes:
        Xn = np.asarray(nodeCoords, dtype=float).reshape(-1, 3)[:, 0]
        ax.plot(Xn, U, "o", markersize=4)

    ax.set_xlabel("x")
    ax.set_ylabel("u_h")
    ax.grid(True)
    return ax


def setup_interactive_figure(xlim=(0.0, 1.0), ylim=None):
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True)
    return fig, ax

def plot_solution_2d(coords_nodes, elements_idx, U, show_mesh=True):
    x = coords_nodes[:, 0]
    y = coords_nodes[:, 1]

    # Keep only the 3 corner nodes of each triangle
    triangles = np.asarray(elements_idx[:, :3], dtype=int)

    tri = mtri.Triangulation(x, y, triangles)

    plt.figure()
    plt.tricontourf(tri, U, levels=30)
    plt.colorbar(label="u_h")

    if show_mesh:
        plt.triplot(tri, color="k", linewidth=0.4, alpha=0.5)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Finite element solution")
    plt.axis("equal")
    plt.show()