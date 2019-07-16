from torchvision.utils import make_grid
import cv2
from PIL import Image
from torchvision import transforms as T
import torch
import numpy as np

transform = []
transform.append(T.Resize(128))
transform.append(T.ToTensor())
transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
transform = T.Compose(transform)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tensor_to_pil(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
    grid = make_grid(denorm(tensor), nrow=nrow, padding=padding, pad_value=pad_value, normalize=normalize, range=range, scale_each=scale_each)
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im


def pil_to_tensor(img):
    img = transform(img)
    img = img.view(1, img.size(0), img.size(1), img.size(2))
    img = img.to(device)
    return img


def cv2_to_pil(cv2im):
    pil_img = cv2.cvtColor(cv2im, cv2.COLOR_BGR2RGB)
    return Image.fromarray(pil_img)


def pil_to_cv2(pil_img):
    cv2im = np.asarray(pil_img)
    return cv2.cvtColor(cv2im, cv2.COLOR_BGR2RGB)


def tensor_to_cv2(tensor):
    return pil_to_cv2(tensor_to_pil(tensor))


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)
