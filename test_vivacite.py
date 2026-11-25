import cv2
import numpy as np
from vivacite import preparer_zone, score_mouvement, SEUIL_VIVACITE

rng = np.random.default_rng(0)
base = cv2.GaussianBlur(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8), (0, 0), 8)
boite = (200, 100, 440, 380)


def bruit(img, sigma=3):
    """Simule le bruit d'une caméra."""
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


# 1. Image fixe (photo) : seul le bruit de caméra change
statique = [bruit(base) for _ in range(3)]
s1 = score_mouvement([preparer_zone(f, boite) for f in statique])
print("Photo statique :", round(s1, 2), "-> vivant ?", s1 > SEUIL_VIVACITE)

# 2. Clignement simulé : une zone s'éclaircit à la 2e image
vivant = []
for k in range(3):
    f = base.copy()
    f[180:230, 260:380] = np.clip(f[180:230, 260:380].astype(int) + (50 if k == 1 else 0), 0, 255)
    vivant.append(bruit(f))
s2 = score_mouvement([preparer_zone(f, boite) for f in vivant])
print("Clignement simulé :", round(s2, 2), "-> vivant ?", s2 > SEUIL_VIVACITE)

# 3. Boîte invalide
print("Boîte trop petite :", preparer_zone(base, (0, 0, 5, 5)))