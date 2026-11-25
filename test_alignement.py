import sys
import cv2
from PIL import Image, ImageOps
from detection import detecter_visage

chemin = sys.argv[1]
img = ImageOps.exif_transpose(Image.open(chemin)).convert("RGB")
tenseur, statut, visage = detecter_visage(img, retourner_image=True)
print(statut)
if visage is not None:
    cv2.imwrite("aligne.jpg", cv2.cvtColor(visage, cv2.COLOR_RGB2BGR))