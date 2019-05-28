import os
import sys

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data.dataset
import torchvision.models.resnet
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import BasicBlock, Bottleneck
import skimage, skimage.io
import numpy as np
import h5py
from tqdm import tqdm

pretrained_model_path = r'/media/wentian/sdb2/work/pretrained_cnn/resnet101-5d3b4d8f.pth'

# not used
model_urls = {
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}

att_size = 14

preprocess = transforms.Compose([
        # trn.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class ResNet(torchvision.models.resnet.ResNet):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__(block, layers, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        fc = x.mean(3).mean(2).squeeze()
        att = F.adaptive_avg_pool2d(x, [att_size, att_size]).squeeze().permute(1, 2, 0)

        return fc, att


def resnet101(pretrained=False):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def extract_image_feature(image_file_list, feature_file):
    print('extracting {} features'.format(len(image_file_list)))
    print('save to {}'.format(feature_file))

    net = resnet101(pretrained=False)
    net.load_state_dict(torch.load(pretrained_model_path))
    net.cuda()
    net.eval()

    h5_file = h5py.File(feature_file, 'w', swmr=True)

    for i, image_file in tqdm(enumerate(image_file_list), total=len(image_file_list), ncols=64):
        I = skimage.io.imread(image_file)
        if len(I.shape) == 2:
            I = I[:, :, np.newaxis]
            I = np.concatenate((I, I, I), axis=2)

        I = I.astype('float32') / 255.0
        I = torch.from_numpy(I.transpose([2, 0, 1])).cuda()
        I = preprocess(I)
        I = I.unsqueeze(0)

        with torch.no_grad():
            fc, att = net(I)

        image_filename = os.path.split(image_file)[-1]
        dataset_fc = h5_file.create_dataset(image_filename, fc.shape, dtype=np.float32)
        dataset_fc[...] = fc

    h5_file.close()


# coco_image_path = ['/home/wentian/work/caption_dataset/MSCOCO/train2014',
#                    '/home/wentian/work/caption_dataset/MSCOCO/val2014']
# all_image = []
# for p in coco_image_path:
#     imgs = os.listdir(p)
#     for i in imgs:
#         all_image.append(os.path.join(p, i))
# extract_image_feature(all_image, '/media/wentian/nvme0n1p5/temp/coco_fc.h5')


flickr30k_image_path = '/media/litong/nvme0n1p5/temp/flickr30k_images'
all_image = [os.path.join(flickr30k_image_path, p) for p in os.listdir(flickr30k_image_path)]
extract_image_feature(all_image, '/media/wentian/nvme0n1p5/temp/flickr30k_fc.h5')