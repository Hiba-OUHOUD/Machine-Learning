import numpy as np
import torch
from PIL import Image
from detection import detecter_visage
from embedding import calculer_embedding

# 1. Détection sur une image noire : aucun visage attendu
img = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
print("Image vide :", detecter_visage(img)[1])

# 2. Embedding sur un tenseur aléatoire : dimension et norme
e = calculer_embedding(torch.randn(3, 160, 160))
print("Dimension :", e.shape)
print("Norme :", round(float(np.linalg.norm(e)), 3))