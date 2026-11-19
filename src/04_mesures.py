from pathlib import Path

import numpy as np
import pandas as pd

# --- Paramètres ---
KEYPOINTS = Path("data/processed/keypoints.csv")
BALLE = Path("data/processed/balle.csv")
SORTIE = Path("data/processed/mesures.csv")

FPS = 30            # toutes les vidéos sont à 30 images/s
CONF_MIN = 0.5      # on n'utilise une articulation que si le modèle en est assez sûr
TROU_MAX = 3        # on tolère 3 images sans balle au milieu d'un lancer
BORD_HAUT = 5       # balle à moins de 5 pixels du haut = sommet probablement hors cadre

kp = pd.read_csv(KEYPOINTS)
balle = pd.read_csv(BALLE)


def milieu(df, a, b, axe):
    return (df[f"{a}_{axe}"] + df[f"{b}_{axe}"]) / 2


def angle(ax, ay, bx, by, cx, cy):
    """Angle au point B (en degrés) formé par les segments BA et BC."""
    v1 = np.stack([ax - bx, ay - by], axis=1)
    v2 = np.stack([cx - bx, cy - by], axis=1)
    cos = (v1 * v2).sum(axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def trajectoire_principale(b):
    """Garde le plus long morceau continu de détections de balle (= le lancer)."""
    images = b["image"].to_numpy()
    coupures = np.where(np.diff(images) > TROU_MAX + 1)[0] + 1
    morceaux = np.split(np.arange(len(b)), coupures)
    plus_long = max(morceaux, key=len)
    return b.iloc[plus_long]


lignes = []
for clip, j in kp.groupby("clip"):
    j = j.sort_values("image").set_index("image")
    b = balle[balle["clip"] == clip].sort_values("image")
    if len(b) < 8:
        print(f"  {clip} : pas assez de positions de balle, ignoré")
        continue

    # --- Unité de mesure : la longueur du buste (milieu des épaules -> milieu des hanches) ---
    # Elle ne dépend ni du zoom ni de la distance à la caméra : on peut comparer les vidéos entre elles.
    fiables = j[(j[["epaule_g_conf", "epaule_d_conf", "hanche_g_conf", "hanche_d_conf"]] > CONF_MIN).all(axis=1)]
    buste = np.hypot(milieu(fiables, "epaule_g", "epaule_d", "x") - milieu(fiables, "hanche_g", "hanche_d", "x"),
                     milieu(fiables, "epaule_g", "epaule_d", "y") - milieu(fiables, "hanche_g", "hanche_d", "y")).median()

    # --- Les moments clés du service ---
    t = trajectoire_principale(b)
    img_debut = int(t["image"].iloc[0])                       # la balle apparaît au-dessus des épaules
    sommet = t.loc[t["balle_y"].idxmin()]                     # point le plus haut (y le plus petit)
    img_sommet = int(sommet["image"])
    sommet_coupe = sommet["balle_y"] < BORD_HAUT

    # Impact : après le sommet, le moment où le poignet le plus haut atteint sa hauteur maximale
    apres = j.loc[img_sommet:img_sommet + 20]
    poignet_haut_y = apres[["poignet_g_y", "poignet_d_y"]].min(axis=1)
    img_impact = int(poignet_haut_y.idxmin())
    y_impact = poignet_haut_y.min()

    # Niveau du sol : position des chevilles pendant la préparation (avant le saut)
    avant = j.loc[:img_debut]
    if len(avant) == 0:
        avant = j.loc[:img_sommet]
    sol_y = milieu(avant, "cheville_g", "cheville_d", "y").median()

    # Flexion des genoux : angle hanche-genou-cheville le plus fermé entre le lancer et l'impact
    phase = j.loc[img_debut:img_impact]
    angles = []
    for cote in ("g", "d"):
        ok = phase[[f"hanche_{cote}_conf", f"genou_{cote}_conf", f"cheville_{cote}_conf"]].gt(CONF_MIN).all(axis=1)
        p = phase[ok]
        if len(p):
            angles.append(angle(p[f"hanche_{cote}_x"].values, p[f"hanche_{cote}_y"].values,
                                p[f"genou_{cote}_x"].values, p[f"genou_{cote}_y"].values,
                                p[f"cheville_{cote}_x"].values, p[f"cheville_{cote}_y"].values))
    flexion = float(np.min(np.concatenate(angles))) if angles else np.nan

    lignes.append({
        "annee": j["annee"].iloc[0],
        "clip": clip,
        # Hauteurs exprimées en "bustes" au-dessus du sol (1 buste ≈ 50-55 cm chez Alcaraz)
        "hauteur_impact": (sol_y - y_impact) / buste,
        "hauteur_sommet_lancer": np.nan if sommet_coupe else (sol_y - sommet["balle_y"]) / buste,
        "sommet_hors_cadre": sommet_coupe,
        # Durées en secondes
        "duree_montee_balle": (img_sommet - img_debut) / FPS,
        "duree_sommet_impact": (img_impact - img_sommet) / FPS,
        "flexion_genoux_deg": flexion,
        "buste_px": buste,
        # Repères utilisés par les scripts de visualisation (06 et 07)
        "img_debut": img_debut,
        "img_sommet": img_sommet,
        "img_impact": img_impact,
        "sol_y": sol_y,
    })

mesures = pd.DataFrame(lignes)
SORTIE.parent.mkdir(parents=True, exist_ok=True)
mesures.to_csv(SORTIE, index=False)
print(f"\n{len(mesures)} services mesurés -> {SORTIE}\n")

# --- Résumé par année : moyenne (régularité = écart-type) ---
# Pour comparer à conditions égales, on écarte les 4 clips de Turin :
# autre angle de caméra, clips trop courts, sommet du lancer souvent hors cadre.
comparables = mesures[~mesures["clip"].str.contains("turin")]
colonnes = ["hauteur_impact", "hauteur_sommet_lancer", "duree_montee_balle", "duree_sommet_impact", "flexion_genoux_deg"]
resume = comparables.groupby("annee")[colonnes].agg(["mean", "std", "count"]).round(2).T
print(resume.to_string())
