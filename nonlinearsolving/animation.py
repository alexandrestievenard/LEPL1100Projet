from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from plot_utils import plot_fe_solution_2d


def save_simulation_animation(saved_fields, saved_times,
                              elemNodeTags, nodeTags, nodeCoords, tag_to_dof,
                              K_FOREST, c_star,
                              add_overlays, make_legend,
                              output_file="invasion_frelon.mp4", fps=5):
    """
    Crée et sauvegarde une animation MP4 à partir des snapshots sauvegardés
    pendant la simulation.

    Paramètres
    ----------
    saved_fields : list of ndarray
        Liste des vecteurs solution U sauvegardés.
    saved_times : list of float
        Temps physiques associés aux snapshots.
    elemNodeTags, nodeTags, nodeCoords, tag_to_dof :
        Données du maillage nécessaires au tracé éléments finis.
    K_FOREST : float
        Valeur maximale utilisée pour l'échelle de couleurs.
    c_star : float
        Vitesse théorique du front.
    add_overlays : callable
        Fonction qui ajoute les annotations géographiques sur la figure.
    make_legend : callable
        Fonction qui construit la légende personnalisée.
    output_file : str
        Nom du fichier vidéo de sortie.
    fps : int
        Nombre d'images par seconde de la vidéo finale.
    """

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d0d1a')

    def update(i):
        ax.clear()
        ax.set_facecolor('#0d0d1a')

        contour = plot_fe_solution_2d(
            elemNodeTags=elemNodeTags,
            nodeTags=nodeTags,
            nodeCoords=nodeCoords,
            U=saved_fields[i],
            tag_to_dof=tag_to_dof,
            show_mesh=False,
            ax=ax,
            vmin=0.0,
            vmax=K_FOREST,
            cmap='plasma'
        )

        add_overlays(ax, saved_times[i])
        make_legend(fig, ax, c_star)

        ax.set_title(
            f'Invasion du Frelon Asiatique — Fisher-KPP\n'
            f't = {saved_times[i]:.1f} an  |  c* = {c_star:.1f} km/an',
            color='white', fontsize=11, pad=10
        )
        ax.set_xlabel('x [km]', color='#aaaacc')
        ax.set_ylabel('y [km]', color='#aaaacc')
        ax.tick_params(colors='#aaaacc')

        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

        ax.axis('equal')
        return contour,

    anim = FuncAnimation(
        fig,
        update,
        frames=len(saved_fields),
        blit=False,
        repeat=False
    )

    writer = PillowWriter(fps=fps)
    anim.save(output_file, writer=writer, dpi=120)
    plt.close(fig)