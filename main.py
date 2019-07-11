import cv2
from util import pil_to_tensor, cv2_to_pil, tensor_to_pil
import numpy as np
import torch
from model import Generator, Classification
from PIL import Image
import os


def classification(image):
    with torch.no_grad():
        out_cls = C(image)
        out_cls = out_cls.view(out_cls.size(1))

        out_cls[out_cls < 0.5] = 0
        out_cls[out_cls >= 0.5] = 1
        return out_cls

def create_labels(c_org):
    hair_color_indices = []
    for i, attr_name in enumerate(selected_attrs):
        if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
            hair_color_indices.append(i)

    out_cls = c_trg.clone()
    for i in range(c_dim):
        if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
            if c_trg[:, i] == 1:
                for j in hair_color_indices:
                    if j != i:
                        out_cls[:, j] = 0
        else:
            out_cls[:, i] = (c_org[i] != c_trg[:, i])
    return out_cls


def trans(img):
    img = img.convert("RGB")
    img_copy = img.copy()
    img_gray = np.asarray(img)
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)

    cascade = cv2.CascadeClassifier(cascade_path)
    facerect = cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=2, minSize=(10, 10))

    for rect in facerect:
        # 顔画像のcrop
        start_x = rect[0] - rect[2]//2
        start_y = rect[1] - rect[3]//2

        img3 = img_copy.copy().crop((start_x, start_y, rect[0] + rect[2] + rect[2]//2, rect[1] + rect[3] + rect[3]//2))

        img3 = pil_to_tensor(img3)
        c_org = classification(img3)
        c = create_labels(c_org)

        face_img = trans_face(img3, c).resize((2 * rect[2], 2 * rect[3]), Image.BICUBIC)

        img.paste(face_img, (start_x, start_y))

    img.save("cascade.jpg")


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def restore_model():
    G.load_state_dict(torch.load("./models/generator.ckpt", map_location=lambda storage, loc: storage))
    C.load_state_dict(torch.load("./models/classification.ckpt", map_location=lambda storage, loc: storage))


def trans_face(x_real, c):
    with torch.no_grad():
        x_real = G(x_real, c)
        return tensor_to_pil(denorm(x_real.data.cpu()), nrow=1, padding=0)


def test():
    print("test start")
    test_data_dir = "./test_data/"
    test_result_dir = "./test_result/"
    test_data = os.listdir(test_data_dir)
    print(test_data)

    for data in test_data:
        img3 = pil_to_tensor(Image.open(test_data_dir + data))
        c_org = classification(img3)
        c = create_labels(c_org)
        img = trans_face(img3, c)

        img.save(test_result_dir + data)
    print("end")


def capture():
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        trans(cv2_to_pil(img))

        key = cv2.waitKey(1)
        if key == 13:
            break
    cap.release()


if __name__ == '__main__':

    cascade_path = "haarcascade_frontalface_alt.xml"

    img_path = "4.jpg"
    input_img = Image.open(img_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = Generator()
    C = Classification()
    G.to(device)
    C.to(device)

    restore_model()

    c_trg = torch.Tensor([[0, 1, 0, 1, 0]]).to(device)
    c_dim = 5
    selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']

    trans(input_img)
    print("complete")

    test()

