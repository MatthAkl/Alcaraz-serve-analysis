from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

# --- Paramètres ---
DOSSIER_VIDEOS = Path("data/raw")
SORTIE = Path("data/processed/keypoints.csv")

# Noms des 17 points articulaires du modèle (format COCO)
POINTS = [
    "nez", "oeil_g", "oeil_d", "oreille_g", "oreille_d",
    "epaule_g", "epaule_d", "coude_g", "coude_d", "poignet_g", "poignet_d",
    "hanche_g", "hanche_d", "genou_g", "genou_d", "cheville_g", "cheville_d",
]
POIGNET_G, POIGNET_D = 9, 10


def traiter_video(chemin):
    """Suit toutes les personnes de la vidéo, puis garde celle qui bouge le plus les bras :
    c'est le serveur. Renvoie une ligne par image avec ses 17 articulations."""
    modele = YOLO("yolo11n-pose.pt")  # rechargé à chaque vidéo pour repartir de zéro dans le suivi

    # 1) Suivi (tracking) : chaque personne garde le même identifiant d'une image à l'autre
    personnes = {}  # identifiant -> liste de (num_image, temps, boite, keypoints)
    nb_images = 0
    fps = None
    for num_image, r in enumerate(modele.track(str(chemin), stream=True, persist=True,
                                               tracker="bytetrack.yaml", verbose=False)):
        nb_images += 1
        if fps is None:
            fps = 30.0
        if r.boxes.id is None:
            continue
        ids = r.boxes.id.int().cpu().numpy()
        boites = r.boxes.xyxy.cpu().numpy()
        kps = r.keypoints.data.cpu().numpy()
        for pid, boite, kp in zip(ids, boites, kps):
            personnes.setdefault(int(pid), []).append((num_image, boite, kp))

    if not personnes:
        return [], nb_images

    # 2) Le serveur = la personne dont les poignets parcourent le plus de distance.
    #    On écarte les personnes coupées par le bord gauche ou droit de l'image
    #    (quelqu'un à moitié visible au bord du cadre "bouge" beaucoup à cause des erreurs de détection).
    largeur_image = r.orig_shape[1]

    def coupee_par_le_bord(suivi):
        au_bord = [b[0] < 3 or b[2] > largeur_image - 3 for _, b, _ in suivi]
        return sum(au_bord) > len(au_bord) / 2

    def mouvement_poignets(suivi):
        poignets = np.array([kp[[POIGNET_G, POIGNET_D], :2] for _, _, kp in suivi])
        if len(poignets) < 2:
            return 0.0
        return float(np.abs(np.diff(poignets, axis=0)).sum())

    candidats = [pid for pid in personnes if not coupee_par_le_bord(personnes[pid])] or list(personnes)
    id_serveur = max(candidats, key=lambda pid: mouvement_poignets(personnes[pid]))
    serveur = {num: (b, kp) for num, b, kp in personnes[id_serveur]}

    # 2 bis) Le suivi perd parfois le joueur et lui donne un nouvel identifiant.
    #        On recolle les morceaux : un autre identifiant qui apparaît au même endroit,
    #        sur des images où le serveur n'était pas suivi, est le même joueur.
    def recouvrement(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0

    for pid in sorted(personnes, key=lambda p: -len(personnes[p])):
        if pid == id_serveur:
            continue
        morceau = personnes[pid]
        if any(num in serveur for num, _, _ in morceau):
            continue  # cette personne est visible en même temps que le serveur : ce n'est pas lui
        debut, fin = morceau[0], morceau[-1]
        proche = min(serveur, key=lambda n: min(abs(n - debut[0]), abs(n - fin[0])))
        bord = debut if abs(proche - debut[0]) < abs(proche - fin[0]) else fin
        if abs(proche - bord[0]) <= 10 and recouvrement(serveur[proche][0], bord[1]) > 0.3:
            for num, b, kp in morceau:
                serveur[num] = (b, kp)

    # 3) Une ligne par image pour le serveur
    lignes = []
    for num_image in sorted(serveur):
        (x1, y1, x2, y2), kp = serveur[num_image]
        ligne = {
            "annee": chemin.parent.name,
            "clip": chemin.stem,
            "image": num_image,
            "temps_s": num_image / fps,
            "boite_x1": x1, "boite_y1": y1, "boite_x2": x2, "boite_y2": y2,
        }
        for nom, (x, y, conf) in zip(POINTS, kp):
            ligne[f"{nom}_x"] = x
            ligne[f"{nom}_y"] = y
            ligne[f"{nom}_conf"] = conf
        lignes.append(ligne)

    return lignes, nb_images


toutes_les_lignes = []
videos = sorted(DOSSIER_VIDEOS.glob("*/*.mp4"))
print(f"{len(videos)} vidéos à traiter")

for chemin in videos:
    lignes, nb_images = traiter_video(chemin)
    toutes_les_lignes.extend(lignes)
    print(f"  {chemin.parent.name}/{chemin.name} : serveur suivi sur {len(lignes)}/{nb_images} images")

df = pd.DataFrame(toutes_les_lignes)
SORTIE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(SORTIE, index=False)
print(f"\n{len(df)} lignes enregistrées dans {SORTIE}")
