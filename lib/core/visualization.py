from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torchvision
import cv2
from utils.transforms import transform_preds
from IPython import embed


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    diff = np.array(
                        [
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)  # 17个最大值的索引
    maxvals = np.amax(heatmaps_reshaped, 2)  # 17个最大值 n*17

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))  # n*17*1

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)  # n*17*2

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def save_batch_image_with_joints(batch_image, batch_joints, file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1], --deleted
    }
    '''

    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            for joint in joints:
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name, normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)  # 64
    heatmap_width = batch_heatmaps.size(3)  # 32

    grid_image = np.zeros((batch_size * heatmap_height,
                           (num_joints + 1) * heatmap_width,
                           3),
                          dtype=np.uint8)  # (64, 576, 3)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255) \
            .clamp(0, 255) \
            .byte() \
            .permute(1, 2, 0) \
            .cpu().numpy()  # (256, 128, 3)
        heatmaps = batch_heatmaps[i].mul(255) \
            .clamp(0, 255) \
            .byte() \
            .cpu().numpy()  # (17, 64, 32)

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))  # (64, 32, 3)
        resized_image_circle = resized_image.copy()
        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)

        for j in range(num_joints):
            cv2.circle(resized_image_circle,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap * 0.7 + resized_image * 0.3
            # draw circle of each keypoint on different images
            cv2.circle(masked_image, (int(preds[i][j][0]), int(preds[i][j][1])), 1, [0, 0, 255], 1)
            width_begin = heatmap_width * (j + 1)
            width_end = heatmap_width * (j + 2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = masked_image

            from PIL import Image, ImageFont, ImageDraw
            text = str(round(maxvals[i][j][0] * 100, 1)) + '|'
            font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 12, encoding="unic")
            x, y = (width_begin, height_begin)
            img_pil = Image.fromarray(grid_image)
            draw = ImageDraw.Draw(img_pil)
            draw.text((x, y), text, font=font)
            grid_image = np.array(img_pil)

            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image_circle

    cv2.imwrite(file_name, grid_image)
