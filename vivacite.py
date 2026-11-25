import cv2
import numpy as np
from detection import mtcnn

TAILLE_ZONE = 64
SEUIL_VIVACITE = 2.0   # différence moyenne de pixels (0-255) ; À CALIBRER avec de vraies captures


def preparer_zone(img_rgb, boite):
    """img_rgb : image numpy RGB. boite : (x1, y1, x2, y2).
    Retourne la zone du visage en gris 64x64 floutée, ou None si la boîte est invalide."""
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in boite]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    gris = cv2.cvtColor(img_rgb[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
    gris = cv2.resize(gris, (TAILLE_ZONE, TAILLE_ZONE), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(gris, (5, 5), 0).astype(np.float32)


def score_mouvement(zones):
    """Différence moyenne de pixels entre zones successives."""
    diffs = [float(np.mean(np.abs(zones[i + 1] - zones[i])))
             for i in range(len(zones) - 1)]
    return float(np.mean(diffs))


def verifier_vivacite(frames, seuil=SEUIL_VIVACITE):
    """frames : liste de 3 images PIL (RGB) consécutives.
    Retourne (vivant, score, statut).
    statut = "ok", "aucun_visage" ou "plusieurs_visages"."""
    milieu = frames[len(frames) // 2]
    boites, _ = mtcnn.detect(milieu)
    if boites is None:
        return False, 0.0, "aucun_visage"
    if len(boites) > 1:
        return False, 0.0, "plusieurs_visages"

    zones = [preparer_zone(np.array(f), boites[0]) for f in frames]
    if any(z is None for z in zones):
        return False, 0.0, "aucun_visage"

    score = score_mouvement(zones)
    return score > seuil, score, "ok"