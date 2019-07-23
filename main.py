import cv2
from util import pil_to_tensor, cv2_to_pil, tensor_to_pil, pil_to_cv2
import numpy as np
import torch
from model import Generator, Classification
from PIL import Image
import os
import time
from deeolab_v3_plus.segmentation import Segmentation

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
        """else:
            if c_trg[:, i] == 1:
                out_cls[:, i] = (c_org[i] != c_trg[:, i])
        """
    return out_cls


def merge_img(img_org, img_trans, img_mask):
    h, w, _ = img_org.shape
    print(img_org.shape)

    img_mask = cv2.bitwise_and(img_trans, img_trans, img_mask)
    img_mask[img_mask==255] = 0
    img_mask = cv2.bitwise_and(img_org, img_org, img_mask)

    return cv2_to_pil(img_mask)


def trans(img):
    img = img.convert("RGB")
    img_copy = img.copy()
    img_gray = np.asarray(img)
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)

    cascade = cv2.CascadeClassifier(cascade_path)
    facerect = cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=2, minSize=(10, 10))

    for rect in facerect:
        start = time.time()

        # 顔画像のcrop
        start_x = rect[0] - rect[2]//2
        start_y = rect[1] - rect[3]//2

        img3 = img_copy.copy().crop((start_x, start_y, rect[0] + rect[2] + rect[2]//2, rect[1] + rect[3] + rect[3]//2))
        img3 = pil_to_tensor(img3)
        c_org = classification(img3)
        c = create_labels(c_org)

        face_img = trans_face(img3, c).resize((2 * rect[2], 2 * rect[3]), Image.LANCZOS)

        cut_img = deeplab.validation(face_img)

        cut_img = cv2.resize(cut_img, (2 * rect[2], 2 * rect[3]), interpolation=cv2.INTER_NEAREST)

        face_img = merge_img(pil_to_cv2(tensor_to_pil(img3.data).resize((2 * rect[2], 2 * rect[3]))), pil_to_cv2(face_img), cut_img)

        img.paste(face_img, (start_x, start_y))
        print(time.time() - start)

    return pil_to_cv2(img)


def restore_model():
    G.load_state_dict(torch.load("./models/generator_5.ckpt", map_location=lambda storage, loc: storage))
    C.load_state_dict(torch.load("./models/classification_5.ckpt", map_location=lambda storage, loc: storage))


def trans_face(x_real, c):
    with torch.no_grad():
        x_real = G(x_real, c)
        return tensor_to_pil(x_real.data.cpu(), nrow=1, padding=0)


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
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video = cv2.VideoWriter("video.mp4", fourcc, 10.0, (640, 640))

    while True:
        _, img = cap.read()
        cv2.imshow("img", img)
        img = trans(cv2_to_pil(img))
        cv2.imshow("trans", img)
        img = cv2.resize(img, (640, 640))
        video.write(img)

        if cv2.waitKey(1) >= 10:
            break
    cap.release()
    video.release()


if __name__ == '__main__':

    cascade_path = "haarcascade_frontalface_alt.xml"

    img_path = "images/img.jpg"
    input_img = Image.open(img_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    c_dim = 5
    deeplab = Segmentation()

    G = Generator(c_dim=c_dim)
    C = Classification(c_dim=c_dim)
    G.to(device)
    C.to(device)

    restore_model()

    selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young', "Eyeglasses", "Smiling"]
    c_trg = torch.Tensor([[0, 1, 0, 1, 1]]).to(device)
    # cv2.imwrite("cascade.jpg", trans(input_img))

    capture()
    # print("complete")

    test()

