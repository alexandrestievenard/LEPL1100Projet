# Simulation Fisher-KPP — Invasion du Frelon Asiatique en Corse

**LEPL1110 — Méthodes des éléments finis**  
**Groupe 92** — Kempeneers Louis, Mouchart Adrien, Stievenard Alexandre  
**Date :** avril 2026

Ce projet simule la propagation spatiale du frelon asiatique (*Vespa velutina*) en Corse à l’aide d’un modèle de réaction-diffusion de type Fisher-KPP sur une géométrie 2D réaliste de l’île.

---

## Prérequis

- Python 3.9 ou supérieur
- Bibliothèques Python :

```bash
pip install numpy scipy matplotlib gmsh shapely pyproj pillow
```

Gmsh doit être installé et accessible dans le `PATH`.

---

## 1. Génération du maillage

Cette étape est à effectuer une seule fois.

```bash
python msh.py
```

Cette commande crée le fichier `invasion_map.msh`, contenant le maillage triangulaire de la Corse avec les groupes physiques suivants :

- `OuterBoundary` : côte de la Corse
- `Mountains` : massifs montagneux

---

## 2. Lancement de la simulation

La simulation se lance avec le script principal :

```bash
python runsimulation.py [OPTIONS]
```

### Options disponibles

| Option | Valeur par défaut | Description |
|---|---:|---|
| `--order` | `1` | Ordre des éléments : `1` pour linéaire P1, `2` pour quadratique P2 |
| `--method` | `imex` | Méthode temporelle : `imex` recommandée ou `newton` |
| `--dt` | `0.1` | Pas de temps en années |
| `--nsteps` | `600` | Nombre de pas de temps, par exemple `600 × 0.1 = 60 ans` |
| `--theta` | `1.0` | Paramètre θ du schéma : `1.0` pour Euler implicite, `0.5` pour Crank-Nicolson |
| `--save_every` | `5` | Sauvegarde un snapshot tous les N pas |
| `--live` | désactivé | Active l’affichage interactif en temps réel |
| `--no_visu` | désactivé | Désactive la génération du GIF final |

---

## Exemples de commandes

### Simulation standard recommandée

```bash
python runsimulation.py --method imex --dt 0.1 --nsteps 600
```

### Simulation avec visualisation en temps réel

```bash
python runsimulation.py --method imex --live
```

### Simulation avec Newton-Raphson

Plus précis, mais plus lent.

```bash
python runsimulation.py --method newton --dt 0.1 --nsteps 400
```

### Simulation avec éléments d’ordre 2

```bash
python runsimulation.py --order 2 --method imex
```

---

## 3. Résultats produits

Le programme génère les résultats suivants :

- `simulation.gif` : animation de l’invasion, sauf si l’option `--no_visu` est utilisée
- Affichage console :
  - évolution de `u_max`
  - fraction du domaine envahie
  - densité moyenne par ville
- Mode `--live` :
  - visualisation interactive pendant le calcul

---

## Structure du projet

| Fichier | Description |
|---|---|
| `msh.py` | Génération du maillage |
| `runsimulation.py` | Point d’entrée principal |
| `gmsh_utils.py` | Interface avec Gmsh |
| `mass.py` | Assemblage de la matrice de masse |
| `stiffness_non_linear.py` | Assemblage de la matrice de rigidité non linéaire |
| `imex_solver.py` | Schéma IMEX : diffusion implicite et réaction explicite |
| `newton_solver.py` | Solveur Newton-Raphson complet |
| `dirichlet.py` | Imposition des conditions de Dirichlet |
| `plot_utils.py` | Visualisation et animation |

---

## Remarques

Le code est entièrement commenté et modulaire.
