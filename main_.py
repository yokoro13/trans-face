import cv2
import os
import numpy as np
import torch
from model import Generator
from torchvision.utils import save_image
from PIL import Image, ImageDraw, ImageFilter
from torchvision import transforms as T
import torchvision.transforms.functional as F

def trans():
    cascade_path = "haarcascade_frontalface_alt.xml"

    img_path = "img.jpg"
    color = (255, 255, 255)

    img2 = cv2.imread(img_path)

    img = Image.open(img_path)
    img = img.convert("RGB")
    img_copy = img.copy()
    img_cv = np.asarray(img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    cascade = cv2.CascadeClassifier(cascade_path)

    facerect = cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

    for rect in facerect:
        # 顔画像のcrop
        img3 = img_copy.copy()
        img3 = img3.crop((rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])).resize((256, 256), Image.BICUBIC)
        # imgs.append(img3)
        img3 = trans_face(img3, rect[0])

        face_img = img3.resize((rect[2], rect[3]), Image.BICUBIC)

        img3.save("./imgs/{}.jpg".format(rect[0]))
        img.paste(face_img, (rect[0], rect[1]))
        cv2.rectangle(img2, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), color=color, thickness=2)

    cv2.imwrite("a.jpg", img2)
    img.save("cascade.jpg")


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def restore_model():
    G.load_state_dict(torch.load("200000-G.ckpt", map_location=lambda storage, loc: storage))


def create_labels(self, c_org):
    """Generate target domain labels for debugging and testing."""
    # Get hair color indices.

    hair_color_indices = []
    for i, attr_name in enumerate(selected_attrs):
        if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
            hair_color_indices.append(i)

    c_trg_list = []
    for i in range(c_dim):

        c_trg = c_org.clone()
        if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
            c_trg[:, i] = 1
            for j in hair_color_indices:
                if j != i:
                    c_trg[:, j] = 0
        else:
            c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.

    c_trg_list.append(c_trg.to(self.device))
    return c_trg_list


def trans_face(img, i):
    with torch.no_grad():
        # Prepare input images and target domain labels.
        x_real = transform(img)
        x_real = x_real.to(device)
        x_real = G(x_real, c_trg)

        x_concat = torch.cat(tuple(x_real), dim=3)
        save_image(denorm(x_real.data.cpu()), "./result/{}.jpg".format(i), nrow=1, padding=0)
        print('Saved real and fake images into {}...'.format(i))
        x_real = x_real.cpu().numpy()
        x_real = np.delete(x_real, 1, axis=1)
        x_real = torch.from_numpy(x_real).to(device)
        print(x_real.shape)
        return to_pil(denorm(x_real.data.cpu()))    # Image.open("./result/{}.jpg".format(i))


if __name__ == '__main__':

    img_path = "img.jpg"
    img = Image.open(img_path)
    img = img.convert("RGB")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = Generator()
    G.to(device)
    restore_model()

    selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']

    transform = T.Compose([T.Resize(256), T.ToTensor()])
    to_pil = T.Compose([T.ToPILImage()])

    c_trg = torch.Tensor([0, 1, 0, 0, 0]).to(device)
    c_org = torch.Tensor([1, 0, 0, 1, 1]).to(device)
    c_dim = 5

    zero_celeba = torch.zeros(1, c_dim).to(device)  # Zero vector for CelebA.

    trans()

