import cv2
import numpy as np
import torch
from model import Generator, Classification
from torchvision.utils import make_grid
from PIL import Image
from torchvision import transforms as T
import os


def classification(image):
    with torch.no_grad():
        out_cls = C(image)
        out_cls = out_cls.view(out_cls.size(1))

        out_cls[out_cls < 0.5] = 0
        out_cls[out_cls >= 0.5] = 1
        return out_cls


def trans():
    cascade_path = "haarcascade_frontalface_alt.xml"

    img_path = "img.jpg"
    color = (255, 255, 255)

    img2 = cv2.imread(img_path)

    img = Image.open(img_path).convert("RGB")
    img_copy = img.copy()
    img_gray = np.asarray(img)
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)

    cascade = cv2.CascadeClassifier(cascade_path)
    facerect = cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

    for rect in facerect:
        # 顔画像のcrop
        start_x = rect[0] - rect[2]//2
        start_y = rect[1] - rect[3]//2

        img3 = img_copy.copy().crop((start_x, start_y, rect[0] + rect[2] + rect[2]//2, rect[1] + rect[3] + rect[3]//2))# .resize((256, 256), Image.BICUBIC)

        img3 = pil_to_tensor(img3)
        # c_org = classification(img3)

        face_img = trans_face(img3).resize((2 * rect[2], 2 * rect[3]), Image.BICUBIC)

        img.paste(face_img, (start_x, start_y))
        cv2.rectangle(img2, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), color=color, thickness=2)

    cv2.imwrite("a.jpg", img2)
    img.save("cascade.jpg")


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def restore_model():
    G.load_state_dict(torch.load("./models/generator.ckpt", map_location=lambda storage, loc: storage))
    C.load_state_dict(torch.load("./models/classification.ckpt", map_location=lambda storage, loc: storage))


def trans_face(x_real):
    with torch.no_grad():
        x_real = G(x_real, c_trg)
        return tensor_to_pil(denorm(x_real.data.cpu()), nrow=1, padding=0)


def tensor_to_pil(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value, normalize=normalize, range=range, scale_each=scale_each)
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im


def pil_to_tensor(img):
    img = transform(img)
    img = img.view(1, img.size(0), img.size(1), img.size(2))
    img = img.to(device)
    return img


def test():
    print("test start")
    test_data_dir = "./test_data/"
    test_result_dir = "./test_result/"
    test_data = os.listdir(test_data_dir)
    print(test_data)
    for data in test_data:
        img = trans_face(pil_to_tensor(Image.open(test_data_dir + data)))
        img.save(test_result_dir + data)
    print("end")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = Generator()
    C = Classification()

    G.to(device)
    C.to(device)
    restore_model()

    c_trg = torch.Tensor([[0, 0, 0, 1, 1]]).to(device)
    c_dim = 5
    selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']

    transform = []
    transform.append(T.Resize(128))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    trans()
    print("complete")

    test()

