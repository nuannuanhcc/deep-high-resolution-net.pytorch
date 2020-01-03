# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images
from IPython import embed

# from torchvision import datasets, transforms
logger = logging.getLogger(__name__)


def validate(config, val_loader, val_dataset, model, output_dir, keypoint_save_path):

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    print('###### The number of samples is {} ######'.format(num_samples))

    with torch.no_grad():
        coors = {}
        for i, input in enumerate(val_loader):
            outputs = model(input[0])
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            #  FLIP_TEST
            if config.TEST.FLIP_TEST:
                print('###### conduct FLIP_TEST ######')
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input[0].cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                              [9, 10], [11, 12], [13, 14], [15, 16]]
                output_flipped = flip_back(output_flipped.cpu().numpy(), flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5
            #  FLIP_TEST

            from core.visualization import get_max_preds, save_batch_image_with_joints, save_batch_heatmaps
            preds, maxvals = get_max_preds(output.clone().cpu().numpy())

            # visualization
            visualization = False
            if visualization:
                prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)
                save_batch_image_with_joints(input[0], preds*4, '{}_pred.jpg'.format(prefix))
                save_batch_heatmaps(input[0], output, '{}_hm_pred.jpg'.format(prefix))

            # normalization
            preds[:, :, 1] = preds[:, :, 1] / output.shape[-2]
            preds[:, :, 0] = preds[:, :, 0] / output.shape[-1]
            preds_maxvals = torch.cat((torch.from_numpy(preds), torch.from_numpy(maxvals)), dim=-1)
            path = input[1]
            for m, n in zip(preds_maxvals, path):
                coors[n] = m
        torch.save(coors, keypoint_save_path)
        print('####### The keypoint extraction is finished #######')
        print('####### The save path is {} #######'.format(keypoint_save_path))


def demo_one_img(config, val_loader, val_dataset, model, output_dir, keypoint_save_path):
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():

        from PIL import Image
        import matplotlib.pyplot as plt
        from torchvision import transforms
        path = '/data/hanchuchu/datasets/PartialREID/partial_body_images/001_001.jpg'
        print('###### extract the key point of img {} ######'.format(path))
        img = Image.open(path).convert('RGB')

        img = transforms.Resize((256, 128), interpolation=3)(img)
        img = transforms.ToTensor()(img)

        input = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img).unsqueeze(0)
        output = model(input)

        if config.TEST.FLIP_TEST:
            print('###### conduct FLIP_TEST ######')
            # this part is ugly, because pytorch has not supported negative index
            # input_flipped = model(input[:, :, :, ::-1])
            input_flipped = np.flip(input.cpu().numpy(), 3).copy()
            input_flipped = torch.from_numpy(input_flipped).cuda()
            outputs_flipped = model(input_flipped)

            if isinstance(outputs_flipped, list):
                output_flipped = outputs_flipped[-1]
            else:
                output_flipped = outputs_flipped

            flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                          [9, 10], [11, 12], [13, 14], [15, 16]]
            output_flipped = flip_back(output_flipped.cpu().numpy(), flip_pairs)
            output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

            # feature is not aligned, shift flipped heatmap for higher accuracy
            if config.TEST.SHIFT_HEATMAP:
                output_flipped[:, :, :, 1:] = \
                    output_flipped.clone()[:, :, :, 0:-1]

            output = (output + output_flipped) * 0.5

        from core.visualization import get_max_preds, save_batch_image_with_joints, save_batch_heatmaps
        preds, maxvals = get_max_preds(output.clone().cpu().numpy())

        visualization = True
        if visualization:
            prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), path.split('/')[-1])
            save_batch_image_with_joints(input, preds * 4, '{}_pred.jpg'.format(prefix))
            save_batch_heatmaps(input, output, '{}_hm_pred.jpg'.format(prefix))
            print('####### The keypoint extraction is finished #######')

