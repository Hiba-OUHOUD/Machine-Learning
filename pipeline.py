import os
import time
from pathlib import Path

import cv2
from PIL import Image

from reconnaissance import reconnaitre, SEUIL_DEFAUT
from vivacite import verifier_vivacite, SEUIL_VIVACITE

PREFIXE = "bibliotheque"
NB_IMAGES = 3          # nombre d'images pour la vivacité
INTERVALLE = 0.25      # secondes entre deux images


def bibliotheque_disponible(prefixe=PREFIXE):
    return (Path(f"{prefixe}_embeddings.npy").exists()
            and Path(f"{prefixe}_ids.npy").exists())


def controler_presence(frames, seuil=SEUIL_DEFAUT, seuil_vivacite=SEUIL_VIVACITE,
                       prefixe=PREFIXE):
    """Fonction à appeler par l'interface.
    frames : liste de 3 images PIL (RGB) consécutives.
    Retourne un dictionnaire :
      decision : "PRESENT", "NON_RECONNU", "USURPATION" ou "VISAGE_NON_EXPLOITABLE"
      nom, distance, vivant, score_vivacite, statut, temps (secondes de traitement)."""
    debut = time.perf_counter()
    res = {"decision": None, "nom": None, "distance": None, "vivant": False,
           "score_vivacite": 0.0, "statut": "ok", "temps": 0.0}

    vivant, score, statut = verifier_vivacite(frames, seuil_vivacite)
    res.update(vivant=vivant, score_vivacite=score, statut=statut)

    if statut != "ok":
        res["decision"] = "VISAGE_NON_EXPLOITABLE"
    elif not vivant:
        res["decision"] = "USURPATION"
    else:
        nom, distance, statut_reco = reconnaitre(frames[len(frames) // 2],
                                                 seuil, prefixe)
        res.update(nom=nom, distance=distance, statut=statut_reco)
        if nom is None:
            res["decision"] = "VISAGE_NON_EXPLOITABLE"
        elif nom == "Non reconnu":
            res["decision"] = "NON_RECONNU"
        else:
            res["decision"] = "PRESENT"

    res["temps"] = time.perf_counter() - debut
    return res


def capturer_images(cap, nb=NB_IMAGES, intervalle=INTERVALLE):
    """Capture nb images espacées de `intervalle` secondes.
    On lit la caméra en continu pour ne pas obtenir d'images anciennes du tampon."""
    frames = []
    for _ in range(nb):
        fin = time.time() + intervalle
        bgr = None
        while time.time() < fin or bgr is None:
            ok, lu = cap.read()
            if not ok:
                return None
            bgr = lu
        frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    return frames


def lancer_webcam(camera=0):
    avec_biblio = bibliotheque_disponible()
    if not avec_biblio:
        print("Bibliothèque introuvable : mode vivacité seule.")
        print("(lancez python bibliotheque.py quand les images seront prêtes)")

    api = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera, api)
    if not cap.isOpened():
        print("Impossible d'ouvrir la webcam.")
        return

    police = cv2.FONT_HERSHEY_SIMPLEX
    message, couleur, jusqu_a = "", (255, 255, 255), 0.0

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        affichage = bgr.copy()
        cv2.putText(affichage, "C = controler | Q = quitter", (10, 25),
                    police, 0.6, (255, 255, 255), 2)
        if time.time() < jusqu_a:
            cv2.putText(affichage, message, (10, 60), police, 0.8, couleur, 2)
        cv2.imshow("Controle d'assiduite", affichage)

        touche = cv2.waitKey(1) & 0xFF
        if touche == ord("q"):
            break
        if touche == ord("c"):
            debut = time.perf_counter()
            frames = capturer_images(cap)
            if frames is None:
                print("Erreur de lecture de la webcam.")
                break

            if avec_biblio:
                r = controler_presence(frames)
                total = time.perf_counter() - debut
                print(r)
                print(f"Temps capture + traitement : {total:.2f} s")
                if r["decision"] == "PRESENT":
                    message, couleur = f"PRESENT : {r['nom']}", (0, 200, 0)
                else:
                    message, couleur = r["decision"], (0, 0, 255)
            else:
                vivant, score, statut = verifier_vivacite(frames)
                print(f"vivant={vivant}  score={score:.2f}  statut={statut}")
                message = f"{'VIVANT' if vivant else 'PHOTO ?'} score={score:.1f}"
                couleur = (0, 200, 0) if vivant else (0, 0, 255)
            jusqu_a = time.time() + 3

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    lancer_webcam()