import numpy as np
from detection import detecter_visage
from embedding import calculer_embedding
from bibliotheque import charger_bibliotheque

SEUIL_DEFAUT = 0.5          # distance cosinus ; sera ajusté avec la courbe ROC
_cache = {}                 # la bibliothèque n'est chargée qu'une seule fois


def distances_cosinus(embedding, bibliotheque):
    """embedding : (512,) normalisé. bibliotheque : (N, 512) normalisée.
    Retourne un tableau de N distances cosinus (0 = identique, ~1 = sans rapport)."""
    return 1.0 - bibliotheque @ embedding


def identifier(embedding, bibliotheque, ids, seuil=SEUIL_DEFAUT):
    """Retourne (nom, distance). nom = "Non reconnu" si la distance minimale
    n'est pas strictement inférieure au seuil."""
    distances = distances_cosinus(embedding, bibliotheque)
    i = int(np.argmin(distances))
    distance = float(distances[i])
    if distance < seuil:
        return str(ids[i]), distance
    return "Non reconnu", distance


def reconnaitre(img, seuil=SEUIL_DEFAUT, prefixe="bibliotheque"):
    """img : image PIL (RGB).
    Retourne (nom, distance, statut).
    statut = "ok", ou la raison de l'échec de détection
    ("aucun_visage", "plusieurs_visages", "confiance_faible")."""
    if prefixe not in _cache:
        _cache[prefixe] = charger_bibliotheque(prefixe)
    bibliotheque, ids = _cache[prefixe]

    visage, statut = detecter_visage(img)
    if visage is None:
        return None, None, statut

    nom, distance = identifier(calculer_embedding(visage), bibliotheque, ids, seuil)
    return nom, distance, "ok"