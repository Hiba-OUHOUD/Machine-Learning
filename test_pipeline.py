import numpy as np
from PIL import Image
from pipeline import controler_presence

noir = [Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8)) for _ in range(3)]
print(controler_presence(noir))