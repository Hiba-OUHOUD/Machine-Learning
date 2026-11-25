from pathlib import Path

import numpy as np

from evaluation import (SEUILS, afficher_resume, balayer_seuils, calculer_distances,
                        choisir_seuil, matrice_confusion, tracer_courbes,
                        tracer_matrice, tracer_roc)

rng = np.random.default_rng(0)


def normaliser(v):
    return v / np.linalg.norm(v)


# Fausse bibliothèque : 6 personnes ; ID_01 et ID_02 se ressemblent volontairement
base = [normaliser(rng.normal(size=512)) for _ in range(6)]
base[1] = normaliser(base[0] + 0.45 * normaliser(rng.normal(size=512)))
bibliotheque = np.array(base)
ids = np.array([f"ID_0{i + 1}" for i in range(6)])

# 10 tentatives par personne : bibliothèque + bruit de force variable
embs, vrais = [], []
for i in range(6):
    for _ in range(10):
        s = rng.uniform(0.5, 2.0)
        embs.append(normaliser(bibliotheque[i] + s * rng.normal(size=512) / np.sqrt(512)))
        vrais.append(i)
# 3 échecs de détection (image sans visage) pour ID_03
for _ in range(3):
    embs.append(np.full(512, np.nan))
    vrais.append(2)
embs, vrais = np.array(embs), np.array(vrais)

D = calculer_distances(embs, bibliotheque)
m = balayer_seuils(D, vrais)

# Vérification de cohérence : chaque tentative est soit bien reconnue,
# soit reconnue comme un autre, soit rejetée -> la somme doit valoir 1
somme = m["accuracy"] + m["far"] + m["frr"]
print("Précision + FAR + FRR = 1 pour tous les seuils :", np.allclose(somme, 1.0))

seuil, regle = choisir_seuil(m)
M = matrice_confusion(D, vrais, seuil, len(ids))
print("Total de la matrice =", M.sum(), "(attendu :", len(embs), ")")

afficher_resume(m, seuil, regle, M, ids, len(embs), 3)

sortie = Path("resultats_test")
sortie.mkdir(exist_ok=True)
tracer_courbes(m, seuil, sortie / "courbes_far_frr.png")
tracer_roc(m, seuil, sortie / "roc.png")
tracer_matrice(M, ids, seuil, sortie / "matrice_confusion.png")
print("Graphiques créés :", sorted(p.name for p in sortie.iterdir()))