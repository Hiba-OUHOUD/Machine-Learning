import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # pas de fenêtre : les graphiques sont enregistrés en PNG
import matplotlib.pyplot as plt
import numpy as np

SEUILS = np.round(np.arange(0.30, 0.801, 0.01), 2)   # plage demandée : 0,3 à 0,8
EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------- données
def embeddings_test(dossier="dataset/test", cache="test_embeddings.npz",
                    recalculer=False):
    """Calcule l'embedding de chaque image de test (un dossier par étudiant).
    Les résultats sont mis en cache pour que les essais de seuils soient instantanés.
    Une image sans visage exploitable donne une ligne de NaN (comptée comme échec)."""
    cache = Path(cache)
    if cache.exists() and not recalculer:
        d = np.load(cache, allow_pickle=False)
        return d["embs"], d["noms"]

    from PIL import Image, ImageOps          # importés ici : inutiles pour les tests
    from detection import detecter_visage
    from embedding import calculer_embedding

    racine = Path(dossier)
    if not racine.exists():
        raise FileNotFoundError(f"Dossier de test introuvable : {dossier}")

    embs, noms = [], []
    for d in sorted(racine.iterdir()):
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in EXTENSIONS:
                continue
            img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
            visage, statut = detecter_visage(img)
            if visage is None:
                print(f"  échec de détection : {d.name}/{f.name} ({statut})")
                embs.append(np.full(512, np.nan, dtype=np.float32))
            else:
                embs.append(calculer_embedding(visage))
            noms.append(d.name)

    if not embs:
        raise ValueError("Aucune image de test trouvée.")
    embs = np.array(embs, dtype=np.float32)
    noms = np.array(noms)
    np.savez(cache, embs=embs, noms=noms)
    return embs, noms


def calculer_distances(embs, bibliotheque):
    """Distance cosinus (M images x N personnes). Image sans visage -> distance infinie."""
    D = 1.0 - embs @ bibliotheque.T
    D[np.isnan(embs).any(axis=1)] = np.inf
    return D


def indices_vrais(noms, ids):
    """Position de la vraie identité dans la bibliothèque (-1 si absente)."""
    index = {str(n): i for i, n in enumerate(ids)}
    return np.array([index.get(str(n), -1) for n in noms])


# ---------------------------------------------------------------- métriques
def metriques(D, vrais, seuil):
    """Définitions du cahier des charges :
    précision = bien reconnu ; FAR = reconnu comme quelqu'un d'autre ;
    FRR = non reconnu (rejeté). Les trois somment à 1."""
    meilleur = np.argmin(D, axis=1)
    dmin = D[np.arange(len(D)), meilleur]
    accepte = dmin < seuil
    correct = accepte & (meilleur == vrais)
    n = len(D)
    return {"seuil": float(seuil),
            "accuracy": correct.sum() / n,
            "far": (accepte & ~correct).sum() / n,
            "frr": (~accepte).sum() / n}


def balayer_seuils(D, vrais, seuils=SEUILS):
    lignes = [metriques(D, vrais, t) for t in seuils]
    return {k: np.array([l[k] for l in lignes])
            for k in ("seuil", "accuracy", "far", "frr")}


def choisir_seuil(m, far_max=0.05, frr_max=0.05):
    """Parmi les seuils qui respectent FAR < 5 % et FRR < 5 %, on prend la meilleure
    précision, au milieu du plateau (seuil plus robuste). Sinon : FAR + FRR minimal."""
    ok = (m["far"] < far_max) & (m["frr"] < frr_max)
    if ok.any():
        cand = np.where(ok)[0]
        meilleurs = cand[m["accuracy"][cand] == m["accuracy"][cand].max()]
        i = int(meilleurs[len(meilleurs) // 2])
        regle = "objectifs FAR < 5 % et FRR < 5 % respectés"
    else:
        i = int(np.argmin(m["far"] + m["frr"]))
        regle = "objectifs NON atteints : seuil minimisant FAR + FRR"
    return float(m["seuil"][i]), regle


def matrice_confusion(D, vrais, seuil, n_personnes):
    """Lignes = identité réelle ; colonnes = personne reconnue, + 1 colonne « Non reconnu »."""
    M = np.zeros((n_personnes, n_personnes + 1), dtype=int)
    meilleur = np.argmin(D, axis=1)
    dmin = D[np.arange(len(D)), meilleur]
    for v, b, dm in zip(vrais, meilleur, dmin):
        M[v, b if dm < seuil else n_personnes] += 1
    return M


# ---------------------------------------------------------------- graphiques
def tracer_courbes(m, seuil, fichier):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(m["seuil"], m["far"] * 100, label="FAR (fausse acceptation)")
    ax.plot(m["seuil"], m["frr"] * 100, label="FRR (faux rejet)")
    ax.plot(m["seuil"], m["accuracy"] * 100, label="Précision")
    ax.axhline(5, color="gray", ls=":", lw=1, label="Objectif FAR/FRR = 5 %")
    ax.axvline(seuil, color="black", ls="--", lw=1, label=f"Seuil retenu = {seuil:.2f}")
    ax.set_xlabel("Seuil (distance cosinus)")
    ax.set_ylabel("%")
    ax.set_title("FAR, FRR et précision selon le seuil")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fichier, dpi=150)
    plt.close(fig)


def tracer_roc(m, seuil, fichier):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(m["far"] * 100, m["accuracy"] * 100, marker=".")
    i = int(np.argmin(np.abs(m["seuil"] - seuil)))
    ax.plot(m["far"][i] * 100, m["accuracy"][i] * 100, "ro",
            label=f"Seuil retenu = {seuil:.2f}")
    ax.set_xlabel("FAR (%)")
    ax.set_ylabel("Taux de reconnaissance correcte (%)")
    ax.set_title("Courbe ROC (seuils de 0,3 à 0,8)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fichier, dpi=150)
    plt.close(fig)


def tracer_matrice(M, ids, seuil, fichier):
    n = len(ids)
    taille = max(6, 0.45 * n + 3)
    fig, ax = plt.subplots(figsize=(taille, taille * 0.9))
    ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(n + 1))
    ax.set_xticklabels(list(ids) + ["Non reconnu"], rotation=90, fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(ids, fontsize=8)
    ax.set_xlabel("Résultat")
    ax.set_ylabel("Identité réelle")
    ax.set_title(f"Matrice de confusion (seuil = {seuil:.2f})")
    if n <= 15:
        for i in range(n):
            for j in range(n + 1):
                ax.text(j, i, M[i, j], ha="center", va="center",
                        color="white" if M[i, j] > M.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(fichier, dpi=150)
    plt.close(fig)


def enregistrer_csv(m, fichier):
    with open(fichier, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seuil", "precision", "FAR", "FRR"])
        for i in range(len(m["seuil"])):
            w.writerow([m["seuil"][i], round(m["accuracy"][i], 4),
                        round(m["far"][i], 4), round(m["frr"][i], 4)])


# ---------------------------------------------------------------- rapport
def afficher_resume(m, seuil, regle, M, ids, n_images, n_echecs):
    print(f"\nImages de test : {n_images}  (échecs de détection : {n_echecs})")
    print(f"\n{'Seuil':>6} {'Précision':>10} {'FAR':>7} {'FRR':>7}")
    for i, t in enumerate(m["seuil"]):
        if round(t * 100) % 5 == 0:
            print(f"{t:6.2f} {m['accuracy'][i]:10.1%} {m['far'][i]:7.1%} {m['frr'][i]:7.1%}")

    i = int(np.argmin(np.abs(m["seuil"] - seuil)))
    acc, far, frr = m["accuracy"][i], m["far"][i], m["frr"][i]
    print(f"\nSeuil retenu : {seuil:.2f}  ({regle})")
    print(f"  Précision {acc:.1%}  (objectif > 90 %)  -> {'OK' if acc > 0.90 else 'NON'}")
    print(f"  FAR       {far:.1%}  (objectif < 5 %)   -> {'OK' if far < 0.05 else 'NON'}")
    print(f"  FRR       {frr:.1%}  (objectif < 5 %)   -> {'OK' if frr < 0.05 else 'NON'}")

    par_personne = np.diag(M[:, :len(ids)]) / np.maximum(M.sum(axis=1), 1)
    pires = np.argsort(par_personne)[:5]
    print("\nPersonnes les moins bien reconnues (à analyser avec l'ingénieur données) :")
    for p in pires:
        print(f"  {ids[p]} : {par_personne[p]:.0%} de reconnaissance correcte")


def main():
    from bibliotheque import charger_bibliotheque

    ap = argparse.ArgumentParser(description="Évaluation du module de reconnaissance")
    ap.add_argument("--dossier", default="dataset/test", help="dossier des images de test")
    ap.add_argument("--recalculer", action="store_true",
                    help="recalculer les embeddings de test (à faire si la détection change)")
    ap.add_argument("--seuil", type=float, default=None,
                    help="imposer un seuil au lieu du seuil recommandé")
    args = ap.parse_args()

    bibliotheque, ids = charger_bibliotheque()
    embs, noms = embeddings_test(args.dossier, recalculer=args.recalculer)

    vrais = indices_vrais(noms, ids)
    inconnus = vrais < 0
    if inconnus.any():
        print(f"Attention : {inconnus.sum()} images ignorées (personne absente de la "
              f"bibliothèque) : {sorted(set(noms[inconnus]))}")
        embs, vrais = embs[~inconnus], vrais[~inconnus]
    if len(embs) == 0:
        raise SystemExit("Aucune image de test utilisable.")

    D = calculer_distances(embs, bibliotheque)
    m = balayer_seuils(D, vrais)
    seuil, regle = choisir_seuil(m)
    if args.seuil is not None:
        seuil, regle = args.seuil, "seuil imposé"

    M = matrice_confusion(D, vrais, seuil, len(ids))
    sortie = Path("resultats")
    sortie.mkdir(exist_ok=True)
    tracer_courbes(m, seuil, sortie / "courbes_far_frr.png")
    tracer_roc(m, seuil, sortie / "roc.png")
    tracer_matrice(M, ids, seuil, sortie / "matrice_confusion.png")
    enregistrer_csv(m, sortie / "metriques.csv")

    afficher_resume(m, seuil, regle, M, ids, len(embs), int(np.isinf(D).all(axis=1).sum()))
    print(f"\nFichiers enregistrés dans le dossier « {sortie} ».")


if __name__ == "__main__":
    main()