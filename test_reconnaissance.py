import numpy as np
from reconnaissance import identifier

rng = np.random.default_rng(0)


def normaliser(v):
    return v / np.linalg.norm(v)


# Fausse bibliothèque : 3 personnes avec des vecteurs aléatoires normalisés
ids = np.array(["ID_01", "ID_02", "ID_03"])
bibliotheque = np.array([normaliser(rng.normal(size=512)) for _ in ids])

# 1. Même personne que ID_02, avec un petit bruit : doit être reconnue
requete = normaliser(bibliotheque[1] + 0.01 * rng.normal(size=512))
print("Même personne :", identifier(requete, bibliotheque, ids))

# 2. Personne inconnue (vecteur aléatoire) : doit être rejetée
inconnu = normaliser(rng.normal(size=512))
print("Inconnu       :", identifier(inconnu, bibliotheque, ids))

# 3. Seuil très strict : même la bonne personne est rejetée
print("Seuil 0.01    :", identifier(requete, bibliotheque, ids, seuil=0.01))