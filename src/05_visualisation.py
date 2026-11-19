from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Paramètres ---
KEYPOINTS = Path("data/processed/keypoints.csv")
BALLE = Path("data/processed/balle.csv")
MESURES = Path("data/processed/mesures.csv")
DOSSIER_VIDEOS = Path("data/raw")
DOSSIER_SORTIE = Path("outputs")

# Un service de chaque année pour les vidéos annotées (modifiable)
CLIPS_VIDEO = ["alcaraz_2023_service_07", "alcaraz_2026_salle_service_01"]

COULEURS = {"2023": "#8a8a8a", "2026": "#e8a33d"}
CONF_MIN = 0.5

# Liaisons entre articulations pour dessiner le squelette (indices COCO)
POINTS = [
    "nez", "oeil_g", "oeil_d", "oreille_g", "oreille_d",
    "epaule_g", "epaule_d", "coude_g", "coude_d", "poignet_g", "poignet_d",
    "hanche_g", "hanche_d", "genou_g", "genou_d", "cheville_g", "cheville_d",
]
OS = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
      (11, 13), (13, 15), (12, 14), (14, 16)]

kp = pd.read_csv(KEYPOINTS)
balle = pd.read_csv(BALLE)
mesures = pd.read_csv(MESURES)
mesures["annee"] = mesures["annee"].astype(str)
DOSSIER_SORTIE.mkdir(exist_ok=True)


# =====================================================================
# 1) Graphique de comparaison 2023 / 2026
# =====================================================================
comparables = mesures[~mesures["clip"].str.contains("turin")]
graphiques = [
    ("duree_montee_balle", "Montée de la balle (s)"),
    ("duree_sommet_impact", "Sommet → impact (s)"),
    ("hauteur_sommet_lancer", "Hauteur du lancer (bustes)"),
    ("flexion_genoux_deg", "Flexion des genoux (°)"),
]

fig, axes = plt.subplots(1, len(graphiques), figsize=(14, 4.4))
for ax, (col, titre) in zip(axes, graphiques):
    etiquettes = []
    for i, annee in enumerate(["2023", "2026"]):
        valeurs = comparables.loc[comparables["annee"] == annee, col].dropna()
        if len(valeurs) < 3:
            # Pas assez de mesures pour conclure (ex. : sommet du lancer hors cadre en 2023)
            etiquettes.append(f"{annee}\nnon mesurable")
            continue
        # Un point par service (légèrement décalé pour qu'ils ne se superposent pas)
        x = i + np.random.default_rng(0).uniform(-0.12, 0.12, len(valeurs))
        ax.scatter(x, valeurs, s=28, color=COULEURS[annee], alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.hlines(valeurs.mean(), i - 0.25, i + 0.25, color="black", linewidth=2)  # trait = moyenne
        # Sous l'année : l'écart-type (σ, plus il est petit, plus le geste est régulier) et le nombre de services
        etiquettes.append(f"{annee}\nσ = {valeurs.std():.2f}  (n={len(valeurs)})")
    ax.set_xticks([0, 1], etiquettes, fontsize=9)
    ax.set_xlim(-0.6, 1.6)
    ax.set_title(titre, fontsize=10, pad=10)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Service d'Alcaraz : 2023 vs 2026 (un point = un service, trait = moyenne)", fontsize=12)
fig.tight_layout()
chemin_graphique = DOSSIER_SORTIE / "comparaison_2023_2026.png"
fig.savefig(chemin_graphique, dpi=150)
print(f"Graphique : {chemin_graphique}")


# =====================================================================
# 2) Vidéo annotée : squelette, trajectoire de la balle, moments clés, mesures
# =====================================================================
def dessiner_squelette(image, ligne, couleur):
    pts = {}
    for i, nom in enumerate(POINTS):
        if ligne[f"{nom}_conf"] > CONF_MIN:
            pts[i] = (int(ligne[f"{nom}_x"]), int(ligne[f"{nom}_y"]))
    for a, b in OS:
        if a in pts and b in pts:
            cv2.line(image, pts[a], pts[b], couleur, 2, cv2.LINE_AA)
    for p in pts.values():
        cv2.circle(image, p, 3, (255, 255, 255), -1, cv2.LINE_AA)


def texte(image, txt, pos, taille=0.5, couleur=(255, 255, 255)):
    cv2.putText(image, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, taille, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, taille, couleur, 1, cv2.LINE_AA)


for clip in CLIPS_VIDEO:
    annee = clip.split("_")[1]
    j = kp[kp["clip"] == clip].set_index("image")
    b = balle[balle["clip"] == clip].set_index("image")
    m = mesures[mesures["clip"] == clip].iloc[0]

    # Moments clés, retrouvés à partir des détections
    img_sommet = int(b["balle_y"].idxmin())
    apres = j.loc[img_sommet:img_sommet + 20]
    img_impact = int(apres[["poignet_g_y", "poignet_d_y"]].min(axis=1).idxmin())

    cap = cv2.VideoCapture(str(DOSSIER_VIDEOS / annee / f"{clip}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    larg, haut = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sortie = DOSSIER_SORTIE / f"{clip}_analyse.mp4"
    writer = cv2.VideoWriter(str(sortie), cv2.VideoWriter_fourcc(*"mp4v"), fps, (larg, haut))

    hex_c = COULEURS[annee].lstrip("#")
    couleur = tuple(int(hex_c[k:k + 2], 16) for k in (4, 2, 0))  # hex -> BGR pour OpenCV

    num = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break

        if num in j.index:
            dessiner_squelette(image, j.loc[num], couleur)

        # Traînée de la balle : toutes les positions déjà vues
        passe = b[b.index <= num]
        pts = [(int(x), int(y)) for x, y in zip(passe["balle_x"], passe["balle_y"])]
        for p1, p2 in zip(pts, pts[1:]):
            cv2.line(image, p1, p2, (0, 255, 255), 1, cv2.LINE_AA)
        if num in b.index:
            cv2.circle(image, pts[-1], 5, (0, 255, 255), 2, cv2.LINE_AA)

        # Étiquettes des moments clés
        if num >= img_sommet and img_sommet in b.index:
            sx, sy = int(b.loc[img_sommet, "balle_x"]), int(b.loc[img_sommet, "balle_y"])
            texte(image, "SOMMET", (sx + 8, max(sy, 12)), 0.4, (0, 255, 255))
        if img_impact <= num < img_impact + 15:
            texte(image, "IMPACT", (larg // 2 - 40, 40), 0.8, (0, 255, 255))

        # Panneau de mesures
        texte(image, f"Alcaraz {annee}", (10, haut - 70), 0.6, couleur)
        texte(image, f"Montee balle : {m['duree_montee_balle']:.2f} s", (10, haut - 48), 0.45)
        texte(image, f"Sommet -> impact : {m['duree_sommet_impact']:.2f} s", (10, haut - 30), 0.45)
        texte(image, f"Flexion genoux : {m['flexion_genoux_deg']:.0f} deg", (10, haut - 12), 0.45)

        writer.write(image)
        num += 1

    cap.release()
    writer.release()
    print(f"Vidéo annotée : {sortie}")
