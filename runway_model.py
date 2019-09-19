import runway

from model import BiSeNet

import torch
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2


@runway.setup(options={'checkpoint': runway.file(extension='.pth')})
def setup(opts):
    net = BiSeNet(n_classes=19)
    net.cuda()
    net.load_state_dict(torch.load(opts['checkpoint']))
    net.eval()
    return net


label_to_id = {
    "background": 0,
    "skin": 1,
    "l_brow": 2,
    "r_brow": 3,
    "l_eye": 4,
    "r_eye": 5,
    "eye_g": 6,
    "l_ear": 7,
    "r_ear": 8,
    "ear_r": 9,
    "nose": 10,
    "mouth": 11,
    "u_lip": 12,
    "l_lip": 13,
    "neck": 14,
    "neck_l": 15,
    "cloth": 16,
    "hair": 17,
    "hat": 18
}

label_to_color = {'background': [0, 0, 0]}

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


@runway.command('parse', inputs={'image': runway.image}, outputs={'parsed_face': runway.segmentation(label_to_id=label_to_id, label_to_color=label_to_color)})
def parse(model, inputs):
    image = inputs['image'].resize((512, 512), Image.BILINEAR)
    img = to_tensor(image)
    img = torch.unsqueeze(img, 0)
    img = img.cuda()
    out = model(img)[0]
    parsing = out.squeeze(0).cpu().detach().numpy().argmax(0)
    return parsing.astype(np.uint8)


if __name__ == '__main__':
    runway.run(model_options={'checkpoint': './79999_iter.pth'})
