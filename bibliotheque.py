import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from detection import detecter_visage
from embedding import calculer_embedding

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def construire_bibliotheque(dossier="dataset/inscription", sortie="bibliotheque"):
    racine = Path(dossier)
    if not racine.exists():
        print(f"Dossier introuvable : {dossier}")
        return

    ids, vecteurs = [], []
    for d in sorted(racine.iterdir()):
        if not d.is_dir():
            continue
        embs, rejets = [], 0
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in EXTENSIONS:
                continue
            # exif_transpose : corrige l'orientation des photos de téléphone
            img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
            visage, statut = detecter_visage(img)
            if visage is None:
                rejets += 1
                print(f"  rejetée : {f.name} ({statut})")
                continue
            embs.append(calculer_embedding(visage))
        if not embs:
            print(f"{d.name} : AUCUNE image exploitable, ignoré")
            continue
        moyen = np.mean(embs, axis=0)
        moyen /= np.linalg.norm(moyen)
        ids.append(d.name)
        vecteurs.append(moyen)
        print(f"{d.name} : {len(embs)} utilisées, {rejets} rejetées")

    if not ids:
        print("Aucune personne enregistrée : bibliothèque non créée.")
        return
    np.save(f"{sortie}_embeddings.npy", np.array(vecteurs))
    np.save(f"{sortie}_ids.npy", np.array(ids))
    print(f"Bibliothèque : {len(ids)} personnes, vecteurs de dimension {len(vecteurs[0])}")


def charger_bibliotheque(prefixe="bibliotheque"):
    return np.load(f"{prefixe}_embeddings.npy"), np.load(f"{prefixe}_ids.npy")


if __name__ == "__main__":
    construire_bibliotheque()