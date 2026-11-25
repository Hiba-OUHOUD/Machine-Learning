import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

device = "cuda" if torch.cuda.is_available() else "cpu"
modele = InceptionResnetV1(pretrained="vggface2").eval().to(device)


@torch.no_grad()
def calculer_embedding(visage):
    """visage : tenseur 3x160x160. Retourne un vecteur numpy (512,) normalisé."""
    emb = modele(visage.unsqueeze(0).to(device))
    emb = F.normalize(emb, p=2, dim=1)
    return emb.squeeze(0).cpu().numpy()