"""Verification de la coherence de l'ensemble d'inscription (role ingenieur donnees).

    python test_coherence.py
    python test_coherence.py --dossier dataset/inscription --aberrant 0.45 --proche 0.6

Trois controles :
  1. Photos aberrantes : distance d'une photo a l'embedding moyen des AUTRES photos
     de la meme personne trop grande (mauvaise photo, autre personne, visage deforme).
  2. Photos peut-etre mal classees : la photo est plus proche de la moyenne d'une
     autre personne que de la sienne.
  3. Personnes trop proches : deux embeddings moyens dont la distance est faible
     (risque de confusion a la reconnaissance).
Le rapport detaille est enregistre dans resultats/coherence.csv.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

EXTENSIONS = {".jpg", ".jpeg", ".png"}
MIN_IMAGES = 20    # cahier des charges : 20 a 25 images par personne


def normaliser(v):
    return v / np.linalg.norm(v)


def charger_embeddings(racine):
    """Retourne {id: (noms_fichiers, embeddings (n,512), rejets [(fichier, statut)])}."""
    from PIL import Image, ImageOps           # importes ici : inutiles pour l'analyse
    from detection import detecter_visage
    from embedding import calculer_embedding

    donnees = {}
    for d in sorted(racine.iterdir()):
        if not d.is_dir():
            continue
        noms, embs, rejets = [], [], []
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in EXTENSIONS:
                continue
            img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
            visage, statut = detecter_visage(img)
            if visage is None:
                rejets.append((f.name, statut))
                continue
            noms.append(f.name)
            embs.append(calculer_embedding(visage))
        embs = np.array(embs) if embs else np.empty((0, 512))
        donnees[d.name] = (noms, embs, rejets)
        print(f"  {d.name} : {len(noms)} utilisables, {len(rejets)} rejetees")
    return donnees


def analyser(donnees, seuil_aberrant, seuil_proche):
    ids = [i for i, (_, e, _) in donnees.items() if len(e) > 0]
    moyens = {i: normaliser(donnees[i][1].mean(axis=0)) for i in ids}

    # 1 et 2 : analyse photo par photo
    lignes = []
    for i in ids:
        noms, embs, _ = donnees[i]
        for k, nom in enumerate(noms):
            if len(embs) >= 2:
                autres = normaliser(np.delete(embs, k, axis=0).mean(axis=0))
                d_propre = float(1.0 - embs[k] @ autres)
            else:
                d_propre = float("nan")
            d_autre, qui = float("inf"), ""
            for o in ids:
                if o != i:
                    d = float(1.0 - embs[k] @ moyens[o])
                    if d < d_autre:
                        d_autre, qui = d, o
            problemes = []
            if d_propre > seuil_aberrant:
                problemes.append("aberrante")
            if d_autre < d_propre or d_autre < seuil_aberrant:
                problemes.append(f"proche de {qui}")
            lignes.append({"id": i, "fichier": nom, "dist_propre": d_propre,
                           "dist_autre_min": d_autre, "autre": qui,
                           "probleme": " + ".join(problemes)})

    # 3 : paires de personnes trop proches
    paires = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            d = float(1.0 - moyens[ids[a]] @ moyens[ids[b]])
            if d < seuil_proche:
                paires.append((ids[a], ids[b], d))
    paires.sort(key=lambda p: p[2])
    return lignes, paires


def rapport(donnees, lignes, paires, seuil_aberrant, seuil_proche):
    print("\n=== Effectifs ===")
    for i, (noms, embs, rejets) in donnees.items():
        alerte = "" if len(noms) >= MIN_IMAGES else f"  <- moins de {MIN_IMAGES} images"
        print(f"  {i} : {len(noms)} utilisables / {len(noms) + len(rejets)}{alerte}")
        for f, statut in rejets:
            print(f"      rejetee : {f} ({statut})")

    propres = [l["dist_propre"] for l in lignes if not np.isnan(l["dist_propre"])]
    if propres:
        print(f"\nDistance photo -> moyenne des autres photos de la personne : "
              f"moyenne {np.mean(propres):.2f}, max {np.max(propres):.2f}")

    suspectes = [l for l in lignes if l["probleme"]]
    print(f"\n=== Photos suspectes (seuil aberrant = {seuil_aberrant}) : {len(suspectes)} ===")
    for l in sorted(suspectes, key=lambda x: -x["dist_propre"] if not np.isnan(x["dist_propre"]) else 0):
        print(f"  {l['id']}/{l['fichier']} : dist propre {l['dist_propre']:.2f}, "
              f"plus proche autre {l['dist_autre_min']:.2f} ({l['autre']}) -> {l['probleme']}")
    if suspectes:
        print("  A verifier visuellement, puis supprimer ou reprendre ces photos.")

    print(f"\n=== Personnes trop proches (distance entre moyennes < {seuil_proche}) : "
          f"{len(paires)} ===")
    for a, b, d in paires:
        print(f"  {a} <-> {b} : {d:.2f}")
    if not paires:
        print("  Aucune : les embeddings moyens sont bien separes.")


def enregistrer_csv(lignes, fichier):
    with open(fichier, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "fichier", "dist_propre", "dist_autre_min", "autre", "probleme"])
        for l in lignes:
            w.writerow([l["id"], l["fichier"], round(l["dist_propre"], 4),
                        round(l["dist_autre_min"], 4), l["autre"], l["probleme"]])


def main():
    ap = argparse.ArgumentParser(description="Coherence de l'ensemble d'inscription")
    ap.add_argument("--dossier", default="dataset/inscription")
    ap.add_argument("--aberrant", type=float, default=0.45,
                    help="distance au-dela de laquelle une photo est suspecte")
    ap.add_argument("--proche", type=float, default=0.6,
                    help="distance sous laquelle deux personnes risquent d'etre confondues")
    args = ap.parse_args()

    racine = Path(args.dossier)
    if not racine.exists():
        raise SystemExit(f"Dossier introuvable : {args.dossier}")

    print("Calcul des embeddings...")
    donnees = charger_embeddings(racine)
    if not any(len(e) for _, e, _ in donnees.values()):
        raise SystemExit("Aucune image exploitable.")

    lignes, paires = analyser(donnees, args.aberrant, args.proche)
    rapport(donnees, lignes, paires, args.aberrant, args.proche)

    sortie = Path("resultats")
    sortie.mkdir(exist_ok=True)
    enregistrer_csv(lignes, sortie / "coherence.csv")
    print(f"\nRapport detaille : {sortie / 'coherence.csv'}")


if __name__ == "__main__":
    main()