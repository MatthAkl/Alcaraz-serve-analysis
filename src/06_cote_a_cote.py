from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# --- Paramètres ---
KEYPOINTS = Path("data/processed/keypoints.csv")
BALLE = Path("data/processed/balle.csv")
MESURES = Path("data/processed/mesures.csv")
DOSSIER_VIDEOS = Path("data/raw")
SORTIE = Path("outputs/cote_a_cote.mp4")

# Les deux services à comparer : (dossier, nom du clip, titre affiché, couleur BGR)
GAUCHE = ("2026", "alcaraz_2026_salle_service_01", "ALCARAZ 2026", (61, 163, 232))
DROITE = ("moi", "matth_service_02", "MATTH", (235, 160, 60))

AVANT_IMPACT = 50   # nombre d'images montrées avant l'impact
APRES_IMPACT = 15   # et après
PANNEAU = (400, 600)  # taille de chaque moitié (largeur, hauteur)
CONF_MIN = 0.5

POINTS = [
    "nez", "oeil_g", "oeil_d", "oreille_g", "oreille_d",
    "epaule_g", "epaule_d", "coude_g", "coude_d", "poignet_g", "poignet_d",
    "hanche_g", "hanche_d", "genou_g", "genou_d", "cheville_g", "cheville_d",
]
OS = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
      (11, 13), (13, 15), (12, 14), (14, 16)]

kp = pd.read_csv(KEYPOINTS)
balle = pd.read_csv(BALLE)
mesures = pd.read_csv(MESURES).set_index("clip")


def angle_genou(ligne):
    """Angle hanche-genou-cheville moyen des deux jambes (None si pas fiable)."""
    angles = []
    for c in ("g", "d"):
        if min(ligne[f"hanche_{c}_conf"], ligne[f"genou_{c}_conf"], ligne[f"cheville_{c}_conf"]) > CONF_MIN:
            a = np.array([ligne[f"hanche_{c}_x"], ligne[f"hanche_{c}_y"]])
            b = np.array([ligne[f"genou_{c}_x"], ligne[f"genou_{c}_y"]])
            d = np.array([ligne[f"cheville_{c}_x"], ligne[f"cheville_{c}_y"]])
            v1, v2 = a - b, d - b
            cos = v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angles.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return np.mean(angles) if angles else None


def preparer(dossier, clip):
    """Lit toutes les images du clip et calcule un cadrage fixe autour du joueur et du lancer."""
    j = kp[kp["clip"] == clip].set_index("image")
    b = balle[balle["clip"] == clip].set_index("image")
    m = mesures.loc[clip]

    cap = cv2.VideoCapture(str(DOSSIER_VIDEOS / dossier / f"{clip}.mp4"))
    images = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        images.append(im)
    cap.release()
    haut_img, larg_img = images[0].shape[:2]

    # Cadrage : du sommet du lancer (avec marge) jusqu'aux pieds, centré sur le joueur
    haut = b["balle_y"].min() - 0.5 * m["buste_px"]
    bas = m["sol_y"] + 0.6 * m["buste_px"]
    hauteur = bas - haut
    largeur = hauteur * PANNEAU[0] / PANNEAU[1]
    centre_x = ((j["boite_x1"] + j["boite_x2"]) / 2).median()
    cadre = (centre_x - largeur / 2, haut, largeur, hauteur)
    return j, b, int(m["img_impact"]), images, cadre


def dessiner(image, j, b, num, couleur):
    """Squelette, traînée de la balle et angle des genoux sur l'image d'origine."""
    if num in j.index:
        ligne = j.loc[num]
        pts = {i: (int(ligne[f"{n}_x"]), int(ligne[f"{n}_y"]))
               for i, n in enumerate(POINTS) if ligne[f"{n}_conf"] > CONF_MIN}
        epaisseur = max(2, image.shape[0] // 250)
        for a, c in OS:
            if a in pts and c in pts:
                cv2.line(image, pts[a], pts[c], couleur, epaisseur, cv2.LINE_AA)
    passe = b[b.index <= num]
    trace = [(int(x), int(y)) for x, y in zip(passe["balle_x"], passe["balle_y"])]
    for p1, p2 in zip(trace, trace[1:]):
        cv2.line(image, p1, p2, (0, 255, 255), max(1, image.shape[0] // 400), cv2.LINE_AA)
    if num in b.index:
        cv2.circle(image, trace[-1], max(4, image.shape[0] // 120), (0, 255, 255), 2, cv2.LINE_AA)


def recadrer(image, cadre):
    """Découpe le cadre (en ajoutant du noir s'il dépasse de l'image) puis redimensionne."""
    x, y, l, h = [int(round(v)) for v in cadre]
    bord = max(l, h)
    grand = cv2.copyMakeBorder(image, bord, bord, bord, bord, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    morceau = grand[y + bord:y + bord + h, x + bord:x + bord + l]
    return cv2.resize(morceau, PANNEAU, interpolation=cv2.INTER_AREA)


def texte(image, txt, pos, taille, couleur=(255, 255, 255)):
    cv2.putText(image, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, taille, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, taille, couleur, 2, cv2.LINE_AA)


cotes = []
for dossier, clip, titre, couleur in (GAUCHE, DROITE):
    j, b, impact, images, cadre = preparer(dossier, clip)
    cotes.append({"j": j, "b": b, "impact": impact, "images": images, "cadre": cadre,
                  "titre": titre, "couleur": couleur, "angle_min": None})

SORTIE.parent.mkdir(exist_ok=True)
writer = cv2.VideoWriter(str(SORTIE), cv2.VideoWriter_fourcc(*"mp4v"), 30, (PANNEAU[0] * 2, PANNEAU[1]))

# On avance image par image, en alignant les deux services sur leur impact
for decalage in range(-AVANT_IMPACT, APRES_IMPACT + 1):
    moities = []
    for c in cotes:
        num = c["impact"] + decalage
        if 0 <= num < len(c["images"]):
            im = c["images"][num].copy()
            dessiner(im, c["j"], c["b"], num, c["couleur"])
            panneau = recadrer(im, c["cadre"])
        else:
            panneau = np.zeros((PANNEAU[1], PANNEAU[0], 3), np.uint8)

        # Flexion des genoux : valeur actuelle et minimum atteint jusqu'ici (avant l'impact)
        if num in c["j"].index and decalage <= 0:
            a = angle_genou(c["j"].loc[num])
            if a is not None:
                c["angle_min"] = a if c["angle_min"] is None else min(c["angle_min"], a)
        texte(panneau, c["titre"], (15, 35), 0.9, c["couleur"])
        if c["angle_min"] is not None:
            texte(panneau, f"Flexion max genoux : {c['angle_min']:.0f} deg", (15, PANNEAU[1] - 20), 0.6)
        if 0 <= decalage < 12:
            texte(panneau, "IMPACT", (PANNEAU[0] // 2 - 60, 80), 1.0, (0, 255, 255))
        moities.append(panneau)

    image_finale = np.hstack(moities)
    cv2.line(image_finale, (PANNEAU[0], 0), (PANNEAU[0], PANNEAU[1]), (255, 255, 255), 2)
    writer.write(image_finale)

writer.release()
print(f"Vidéo côte à côte : {SORTIE}")
