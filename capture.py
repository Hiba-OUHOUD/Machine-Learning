"""Capture de photos par webcam pour constituer la base d'images.

Exemples :
    python capture.py                                # le script demande l'identifiant, etudiant apres etudiant
    python capture.py ID_01                          # 25 photos d'inscription
    python capture.py ID_01 --ensemble test --nombre 10
    python capture.py ID_01 --auto                   # capture automatique (touche A pour pause)

Touches : ESPACE = prendre une photo | A = auto on/off | Q = quitter

Chaque photo est verifiee avec le meme detecteur que le reste du projet :
elle n'est enregistree que si exactement un visage net est detecte ET si la
qualite est suffisante (luminosite, nettete, taille du visage). Option --sans-qualite
pour desactiver ce controle si les seuils sont trop stricts pour votre webcam.
Les photos vont dans dataset/<ensemble>/<identifiant>/ (un dossier par etudiant),
ce que lisent bibliotheque.py (inscription) et evaluation.py (test).
"""
import argparse
import os
import re
import time
from pathlib import Path

import cv2
from PIL import Image

from detection import detecter_visage, mtcnn

DOSSIERS = {"inscription": "dataset/inscription", "test": "dataset/test"}
# Cahier des charges : 20-25 images/personne (inscription), 5-10 (test)
MINIMUM = {"inscription": 20, "test": 5}

# Consignes affichees a l'ecran pour varier les photos (sans accents : police OpenCV)
CONSIGNES = [
    "Face, expression neutre",
    "Souriez",
    "Tete legerement tournee a gauche",
    "Tete legerement tournee a droite",
    "Changez d'eclairage / rapprochez-vous",
]

# Controle de qualite (a ajuster si trop strict ou trop permissif pour votre webcam)
LUM_MIN, LUM_MAX = 70, 190     # luminosite moyenne du visage (0-255)
NETTETE_MIN = 50.0             # variance du Laplacien (plus petit = plus flou)
TAILLE_MIN = 0.18              # largeur du visage / largeur de l'image

REJETS = {
    "aucun_visage": "Aucun visage detecte",
    "plusieurs_visages": "Plusieurs visages : une seule personne",
    "confiance_faible": "Visage pas assez net / mal eclaire",
}


def verifier_qualite(bgr, boite):
    """bgr : image OpenCV. boite : (x1, y1, x2, y2) du visage.
    Retourne None si la qualite est bonne, sinon un message (sans accents) pour l'ecran."""
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in boite]
    if (x2 - x1) / w < TAILLE_MIN:
        return "Visage trop loin : rapprochez-vous"
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return "Visage mal cadre"
    zone = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    zone = cv2.resize(zone, (160, 160), interpolation=cv2.INTER_AREA)
    luminosite = float(zone.mean())
    if luminosite < LUM_MIN:
        return "Trop sombre : ajoutez de la lumiere de face"
    if luminosite > LUM_MAX:
        return "Trop eclaire : evitez reflet et contre-jour"
    if cv2.Laplacian(zone, cv2.CV_64F).var() < NETTETE_MIN:
        return "Image floue : restez immobile"
    return None


def prochain_numero(dossier, identifiant):
    """Premier numero libre, pour ne jamais ecraser une photo existante."""
    n = 1
    while (dossier / f"{identifiant}_{n:02d}.jpg").exists():
        n += 1
    return n


def capturer(identifiant, ensemble, nombre, camera, auto, delai, qualite=True):
    dossier = Path(DOSSIERS[ensemble]) / identifiant
    dossier.mkdir(parents=True, exist_ok=True)
    numero = prochain_numero(dossier, identifiant)
    # L'objectif porte sur le TOTAL du dossier (cahier : 20 a 25 par personne en inscription)
    deja = sum(1 for f in dossier.iterdir()
               if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if deja >= nombre:
        print(f"{identifiant} : deja {deja} photos (objectif {nombre}), rien a faire.")
        return

    api = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera, api)
    if not cap.isOpened():
        print("Impossible d'ouvrir la webcam.")
        return

    police = cv2.FONT_HERSHEY_SIMPLEX
    enregistrees = 0
    auto_actif = auto
    prochain_auto = time.time() + delai
    message, couleur, jusqu_a = "", (255, 255, 255), 0.0

    while deja + enregistrees < nombre:
        ok, bgr = cap.read()
        if not ok:
            print("Erreur de lecture de la webcam.")
            break

        consigne = CONSIGNES[min((deja + enregistrees) * len(CONSIGNES) // nombre,
                                 len(CONSIGNES) - 1)]
        if ensemble == "test":     # photos de test : conditions reelles, sans pose
            consigne = "Comportement naturel (conditions reelles)"
        affichage = bgr.copy()
        cv2.putText(affichage, f"{identifiant} [{ensemble}] : {deja + enregistrees}/{nombre}",
                    (10, 25), police, 0.7, (255, 255, 255), 2)
        cv2.putText(affichage, consigne, (10, 55), police, 0.7, (0, 255, 255), 2)
        etat = "AUTO" if auto_actif else "MANUEL"
        cv2.putText(affichage, f"{etat} | ESPACE = photo | A = auto | Q = quitter",
                    (10, affichage.shape[0] - 12), police, 0.5, (255, 255, 255), 1)
        if time.time() < jusqu_a:
            cv2.putText(affichage, message, (10, 90), police, 0.8, couleur, 2)
        cv2.imshow("Capture des photos", affichage)

        touche = cv2.waitKey(1) & 0xFF
        if touche == ord("q"):
            break
        if touche == ord("a"):
            auto_actif = not auto_actif
            prochain_auto = time.time() + delai

        if touche == 32 or (auto_actif and time.time() >= prochain_auto):
            prochain_auto = time.time() + delai
            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            visage, statut = detecter_visage(img)
            if visage is None:
                message, couleur = REJETS.get(statut, statut), (0, 0, 255)
            else:
                probleme = None
                if qualite:
                    boites, _ = mtcnn.detect(img)
                    if boites is not None and len(boites) == 1:
                        probleme = verifier_qualite(bgr, boites[0])
                if probleme:
                    message, couleur = probleme, (0, 0, 255)
                else:
                    chemin = dossier / f"{identifiant}_{numero:02d}.jpg"
                    cv2.imwrite(str(chemin), bgr)   # image entiere : bibliotheque.py refait la detection
                    numero += 1
                    enregistrees += 1
                    message, couleur = f"Enregistree ({deja + enregistrees}/{nombre})", (0, 200, 0)
            jusqu_a = time.time() + 1.5

    cap.release()
    cv2.destroyAllWindows()
    total = deja + enregistrees
    print(f"{enregistrees} photo(s) enregistree(s) ; {total} au total dans {dossier}")
    minimum = MINIMUM[ensemble]
    if total < minimum:
        print(f"ATTENTION : {total} photos seulement, le cahier des charges demande "
              f"au moins {minimum} ({ensemble}). Relancez la capture pour completer.")
    if ensemble == "inscription" and enregistrees:
        print("Pensez a relancer : python bibliotheque.py")


def session(identifiant, args, nombre):
    """Un etudiant : verification de l'identifiant, consentement, puis capture."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", identifiant):
        print("Identifiant invalide : lettres, chiffres, _ et - seulement.")
        return
    # Exigence du cahier des charges : consentement ecrit avant toute collecte
    rep = input(f"{identifiant} : le formulaire de consentement est-il signe ? (o/n) ")
    if rep.strip().lower() not in ("o", "oui"):
        print("Collecte annulee : consentement requis.")
        return
    capturer(identifiant, args.ensemble, nombre, args.camera, args.auto, args.delai,
             qualite=not args.sans_qualite)


def main():
    ap = argparse.ArgumentParser(description="Capture de photos par webcam")
    ap.add_argument("identifiant", nargs="?", default=None,
                    help="identifiant de l'etudiant (ex. ID_01) ; sans identifiant, "
                         "le script le demande, etudiant apres etudiant")
    ap.add_argument("--ensemble", choices=DOSSIERS, default="inscription")
    ap.add_argument("--nombre", type=int, default=None,
                    help="nombre de photos (defaut : 25 inscription, 10 test)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--auto", action="store_true", help="capture automatique")
    ap.add_argument("--sans-qualite", action="store_true",
                    help="desactiver le controle de qualite (luminosite, nettete, taille)")
    ap.add_argument("--delai", type=float, default=1.5,
                    help="secondes entre deux photos en mode auto")
    args = ap.parse_args()

    nombre = args.nombre or (25 if args.ensemble == "inscription" else 10)
    if args.ensemble == "test":
        print("Rappel : photos de test = autre moment / autre lieu que l'inscription.")

    if args.identifiant:                 # un seul etudiant, identifiant donne
        session(args.identifiant, args, nombre)
        return

    while True:                          # mode interactif : un etudiant apres l'autre
        identifiant = input("\nIdentifiant de l'etudiant (vide pour terminer) : ").strip()
        if not identifiant:
            break
        session(identifiant, args, nombre)


if __name__ == "__main__":
    main()
    