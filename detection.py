import math
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(keep_all=True, device=device)

TAILLE = 160
# Position cible des yeux dans l'image 160x160 (fractions de la taille).
# À ajuster si le visage paraît trop serré ou trop large.
OEIL_GAUCHE = (0.33, 0.38)
OEIL_DROIT = (0.67, 0.38)


def aligner_visage(img_rgb, points):
    """img_rgb : image numpy RGB. points : tableau (5, 2) de MTCNN.
    Retourne le visage aligné, 160x160 RGB (uint8)."""
    oeil_g, oeil_d = points[0], points[1]

    dx, dy = oeil_d[0] - oeil_g[0], oeil_d[1] - oeil_g[1]
    angle = math.degrees(math.atan2(dy, dx))

    dist_actuelle = math.hypot(dx, dy)
    dist_cible = (OEIL_DROIT[0] - OEIL_GAUCHE[0]) * TAILLE
    echelle = dist_cible / dist_actuelle

    centre = ((oeil_g[0] + oeil_d[0]) / 2.0, (oeil_g[1] + oeil_d[1]) / 2.0)
    M = cv2.getRotationMatrix2D(centre, angle, echelle)

    # Placer le centre des yeux à sa position cible dans l'image de sortie
    cible_x = (OEIL_GAUCHE[0] + OEIL_DROIT[0]) / 2.0 * TAILLE
    cible_y = OEIL_GAUCHE[1] * TAILLE
    M[0, 2] += cible_x - centre[0]
    M[1, 2] += cible_y - centre[1]

    return cv2.warpAffine(img_rgb, M, (TAILLE, TAILLE),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def vers_tenseur(visage_rgb):
    """Normalisation attendue par FaceNet : (x - 127.5) / 128."""
    t = torch.from_numpy(visage_rgb).permute(2, 0, 1).float()
    return (t - 127.5) / 128.0


def detecter_visage(img, seuil_confiance=0.90, retourner_image=False):
    """img : image PIL (RGB).
    Retourne (tenseur 3x160x160, "ok") ou (None, raison).
    Si retourner_image=True : (tenseur, statut, visage_aligné_numpy)."""
    def sortie(tenseur, statut, visage=None):
        return (tenseur, statut, visage) if retourner_image else (tenseur, statut)

    boites, probas, points = mtcnn.detect(img, landmarks=True)
    if boites is None:
        return sortie(None, "aucun_visage")
    if len(boites) > 1:
        return sortie(None, "plusieurs_visages")
    if probas[0] < seuil_confiance:
        return sortie(None, "confiance_faible")

    visage = aligner_visage(np.array(img), points[0])
    return sortie(vers_tenseur(visage), "ok", visage)