from pathlib import Path

import matplotlib.pyplot as plt
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)  # moyennes de colonnes vides

# --- Paramètres ---
KEYPOINTS = Path("data/processed/keypoints.csv")
BALLE = Path("data/processed/balle.csv")
MESURES = Path("data/processed/mesures.csv")
SORTIE = Path("outputs/courbes_geste.png")

FPS = 30
AVANT, APRES = 45, 10   # fenêtre affichée, en images autour de l'impact
CONF_MIN = 0.5
GROUPES = {  # nom affiché -> (filtre sur les clips, couleur)
    "Alcaraz 2023": (lambda c: "2023" in c, "#8a8a8a"),
    "Alcaraz 2026": (lambda c: "2026_salle" in c, "#e8a33d"),
    "Matth": (lambda c: c.startswith("matth"), "#3c78d8"),
}

kp = pd.read_csv(KEYPOINTS)
balle = pd.read_csv(BALLE)
mesures = pd.read_csv(MESURES).set_index("clip")


def angle_genoux(df):
    """Angle hanche-genou-cheville moyen des deux jambes, image par image."""
    angles = []
    for c in ("g", "d"):
        ok = df[[f"hanche_{c}_conf", f"genou_{c}_conf", f"cheville_{c}_conf"]].gt(CONF_MIN).all(axis=1)
        v1 = np.stack([df[f"hanche_{c}_x"] - df[f"genou_{c}_x"], df[f"hanche_{c}_y"] - df[f"genou_{c}_y"]], 1)
        v2 = np.stack([df[f"cheville_{c}_x"] - df[f"genou_{c}_x"], df[f"cheville_{c}_y"] - df[f"genou_{c}_y"]], 1)
        cos = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
        a = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        angles.append(np.where(ok, a, np.nan))
    return np.nanmean(np.stack(angles), axis=0)


def courbes_du_clip(clip):
    """Trois courbes alignées sur l'impact (temps 0) : poignet le plus haut, genoux, balle."""
    m = mesures.loc[clip]
    j = kp[kp["clip"] == clip].set_index("image")
    b = balle[balle["clip"] == clip].set_index("image")
    images = np.arange(m["img_impact"] - AVANT, m["img_impact"] + APRES + 1)

    j = j.reindex(images)  # images manquantes -> NaN
    poignet = (m["sol_y"] - j[["poignet_g_y", "poignet_d_y"]].min(axis=1)) / m["buste_px"]
    genoux = angle_genoux(j)
    hauteur_balle = (m["sol_y"] - b["balle_y"].reindex(images)) / m["buste_px"]
    return poignet.to_numpy(), genoux, hauteur_balle.to_numpy()


temps = np.arange(-AVANT, APRES + 1) / FPS
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
titres = ["Hauteur du poignet le plus haut (bustes)", "Angle des genoux (°) — plus bas = plus fléchi",
          "Hauteur de la balle (bustes)"]

for nom, (filtre, couleur) in GROUPES.items():
    clips = [c for c in mesures.index if filtre(c)]
    if not clips:
        continue
    series = [courbes_du_clip(c) for c in clips]
    for k, ax in enumerate(axes):
        donnees = np.stack([s[k] for s in series])
        # On n'affiche un point que si au moins la moitié des services ont une valeur à cet instant
        assez = np.sum(~np.isnan(donnees), axis=0) >= max(2, len(clips) / 2)
        with np.errstate(all="ignore"):
            moyenne = np.where(assez, np.nanmean(donnees, axis=0), np.nan)
            ecart = np.where(assez, np.nanstd(donnees, axis=0), np.nan)
        ax.plot(temps, moyenne, color=couleur, linewidth=2.2, label=f"{nom} (n={len(clips)})")
        ax.fill_between(temps, moyenne - ecart, moyenne + ecart, color=couleur, alpha=0.15)

for ax, titre in zip(axes, titres):
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(titre, fontsize=10, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False, fontsize=9)
axes[0].text(0.005, axes[0].get_ylim()[1], " impact", fontsize=8, va="top")
axes[-1].set_xlabel("Temps avant / après l'impact (s)")
fig.suptitle("Le geste du service, aligné sur l'impact (trait = moyenne, zone = ± écart-type)", fontsize=12)
fig.tight_layout()
SORTIE.parent.mkdir(exist_ok=True)
fig.savefig(SORTIE, dpi=150)
print(f"Courbes : {SORTIE}")
