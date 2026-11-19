from pathlib import Path

import cv2
from ultralytics import YOLO

# --- Paramètres ---
VIDEO = Path("data/raw/2026/alcaraz_2026_salle_service_01.mp4")
SORTIE = Path("outputs") / f"{VIDEO.stem}_pose.mp4"

# Modèle de pose pré-entraîné (téléchargé automatiquement la première fois)
modele = YOLO("yolo11n-pose.pt")

# --- Ouverture de la vidéo ---
cap = cv2.VideoCapture(str(VIDEO))
fps = cap.get(cv2.CAP_PROP_FPS)
largeur = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
hauteur = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Vidéo : {largeur}x{hauteur}, {fps:.0f} images/s")

SORTIE.parent.mkdir(exist_ok=True)
writer = cv2.VideoWriter(str(SORTIE), cv2.VideoWriter_fourcc(*"mp4v"), fps, (largeur, hauteur))

num_image = 0
images_avec_joueur = 0

while True:
    ok, image = cap.read()
    if not ok:
        break

    resultat = modele(image, verbose=False)[0]

    # Si plusieurs personnes sont détectées, on garde la plus grande (le joueur filmé)
    if len(resultat.boxes) > 0:
        boites = resultat.boxes.xyxy.cpu().numpy()
        surfaces = (boites[:, 2] - boites[:, 0]) * (boites[:, 3] - boites[:, 1])
        idx = int(surfaces.argmax())
        resultat = resultat[idx : idx + 1]
        images_avec_joueur += 1

    # Dessine la boîte et le squelette sur l'image
    image_annotee = resultat.plot()
    writer.write(image_annotee)
    num_image += 1

cap.release()
writer.release()

print(f"{num_image} images traitées, joueur détecté sur {images_avec_joueur}")
print(f"Vidéo annotée : {SORTIE}")
