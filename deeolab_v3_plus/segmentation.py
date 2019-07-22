from deeolab_v3_plus.modeling.sync_batchnorm.replicate import patch_replication_callback
from deeolab_v3_plus.modeling.deeplab import *
from deeolab_v3_plus.modeling.loss import SegmentationLosses
import torch
import torch.nn
import numpy as np
from torchvision import transforms as T


def transform_val(sample):
    composed_transforms = T.Compose([
        T.Resize(128),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return composed_transforms(sample)


class Segmentation(object):
    def __init__(self):

        # Define network
        model = DeepLab(num_classes=21,
                        backbone="resnet",
                        output_stride=16,
                        sync_bn=True,
                        freeze_bn=False)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': 0.007},
                        {'params': model.get_10x_lr_params(), 'lr': 0.007 * 10}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=0.9,
                                    weight_decay=5e-4, nesterov=False)

        # Define Criterion
        # whether to use class balanced weights
        self.criterion = SegmentationLosses(weight=None, cuda=True).build_loss(mode="ce")
        self.model, self.optimizer = model, optimizer

        # Using cuda
        self.model = torch.nn.DataParallel(self.model, device_ids=[0])
        patch_replication_callback(self.model)
        self.model = self.model.cuda()

        self.model.module.load_state_dict(torch.load("models/deeplab.ckpt", map_location=lambda storage, loc: storage))

    def validation(self, img):
        self.model.eval()
        image = transform_val(img).unsqueeze(0).cuda()
        with torch.no_grad():
            output = self.model(image)
            result = output.data.cpu().numpy()
            prediction = np.argmax(result, axis=1)
            result = prediction[0]
            h, w = result.shape
            rgb = np.empty((h, w, 3), dtype=np.uint8)
            rgb[:, :, 0] = result
            rgb[:, :, 1] = result
            rgb[:, :, 2] = result
            rgb[rgb != 0] = 255
            return rgb
