import time
import numpy as np
import torch
from PIL import Image
from detection import detecter_visage
from embedding import calculer_embedding

img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
visage = torch.randn(3, 160, 160)

# Échauffement : le premier appel est toujours plus lent
detecter_visage(img)
calculer_embedding(visage)

N = 10
debut = time.perf_counter()
for _ in range(N):
    detecter_visage(img)
t_det = (time.perf_counter() - debut) / N

debut = time.perf_counter()
for _ in range(N):
    calculer_embedding(visage)
t_emb = (time.perf_counter() - debut) / N

print(f"Détection MTCNN   : {t_det:.3f} s")
print(f"Embedding FaceNet : {t_emb:.3f} s")
# Un contrôle = 2 détections (vivacité + reconnaissance) + 1 embedding + ~0,5 s de capture
print(f"Estimation d'un contrôle complet : {2 * t_det + t_emb + 0.5:.2f} s")