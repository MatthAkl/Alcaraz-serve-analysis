from collections import Counter
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

# --- Paramètres ---
DOSSIER_VIDEOS = Path("data/raw")
KEYPOINTS = Path("data/processed/keypoints.csv")
SORTIE = Path("data/processed/balle.csv")

CLASSE_BALLE = 32       # "sports ball" dans les classes COCO
CONFIANCE_MIN = 0.1     # seuil bas : la balle est petite et souvent floue
IMAGES_IMMOBILE = 10    # une "balle" vue au même endroit sur 10 images ou plus = balle posée, on l'ignore

# Modèle de détection d'objets (un peu plus précis que la version "n")
modele = YOLO("yolo11s.pt")

# Positions du serveur, calculées à l'étape 3
kp = pd.read_csv(KEYPOINTS).set_index(["clip", "image"])


def dans_zone_lancer(bx, by, joueur):
    """La balle du lancer est au-dessus des épaules du joueur et pas trop loin de lui horizontalement."""
    largeur = joueur["boite_x2"] - joueur["boite_x1"]
    hauteur = joueur["boite_y2"] - joueur["boite_y1"]
    centre_joueur = (joueur["boite_x1"] + joueur["boite_x2"]) / 2
    epaules_y = min(joueur["epaule_g_y"], joueur["epaule_d_y"])
    return by < epaules_y and abs(bx - centre_joueur) < largeur + 0.5 * hauteur


lignes = []
videos = sorted(DOSSIER_VIDEOS.glob("*/*.mp4"))
print(f"{len(videos)} vidéos à traiter")

for chemin in videos:
    # 1) On relève toutes les balles candidates, image par image
    candidates = []  # (num_image, x, y, confiance)
    cap = cv2.VideoCapture(str(chemin))
    num_image = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if (chemin.stem, num_image) in kp.index:
            joueur = kp.loc[(chemin.stem, num_image)]
            # imgsz=1280 : on agrandit l'image pour mieux voir les petits objets
            r = modele(image, imgsz=1280, conf=CONFIANCE_MIN, classes=[CLASSE_BALLE], verbose=False)[0]
            for (x1, y1, x2, y2), conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                bx, by = (x1 + x2) / 2, (y1 + y2) / 2
                if dans_zone_lancer(bx, by, joueur):
                    candidates.append((num_image, bx, by, conf))
        num_image += 1
    cap.release()

    # 2) On écarte les balles immobiles (posées au fond du court) :
    #    on découpe l'image en cases de 5 pixels et on compte combien de fois chaque case revient
    case = lambda x, y: (round(x / 5), round(y / 5))
    frequence = Counter(case(x, y) for _, x, y, _ in candidates)
    en_mouvement = [c for c in candidates if frequence[case(c[1], c[2])] < IMAGES_IMMOBILE]

    # 3) S'il reste plusieurs balles sur une même image, on garde la plus sûre
    meilleures = {}
    for num, x, y, conf in en_mouvement:
        if num not in meilleures or conf > meilleures[num][3]:
            meilleures[num] = (num, x, y, conf)

    for num, x, y, conf in sorted(meilleures.values()):
        lignes.append({
            "annee": chemin.parent.name,
            "clip": chemin.stem,
            "image": num,
            "balle_x": x,
            "balle_y": y,
            "balle_conf": conf,
        })
    print(f"  {chemin.parent.name}/{chemin.name} : balle trouvée sur {len(meilleures)}/{num_image} images")

df = pd.DataFrame(lignes)
SORTIE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(SORTIE, index=False)
print(f"\n{len(df)} positions de balle enregistrées dans {SORTIE}")
