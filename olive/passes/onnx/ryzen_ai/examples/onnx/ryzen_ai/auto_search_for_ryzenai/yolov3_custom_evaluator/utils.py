#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import numpy as np
import cv2
import torch
import time
import torchvision
import random


def box_iou(box1, box2):
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)


def plot_one_box(x, image, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def clip_coords(boxes, img_shape):
    boxes[:, 0].clamp_(0, img_shape[1])
    boxes[:, 1].clamp_(0, img_shape[0])
    boxes[:, 2].clamp_(0, img_shape[1])
    boxes[:, 3].clamp_(0, img_shape[0])


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):

    if ratio_pad is None:
        gain = max(img1_shape) / max(img0_shape)
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def xywh2xyxy(x):
    y = torch.zeros_like(x) if isinstance(
        x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def letterbox(img, new_shape=(416, 416), color=(114, 114, 114), auto=True,
              scaleFill=False, scaleup=True):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def non_max_suppression(
        prediction,
        conf_thres=0.1,
        iou_thres=0.6,
        multi_label=True,
        classes=None,
        agnostic=False):

    merge = True
    min_wh, max_wh = 2, 4096
    time_limit = 10.0

    t = time.time()
    nc = prediction[0].shape[1] - 5
    multi_label &= nc > 1
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[x[:, 4] > conf_thres]
        x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]

        if not x.shape[0]:
            continue

        x[..., 5:] *= x[..., 4:5]

        box = xywh2xyxy(x[:, :4])

        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero().t()
            x = torch.cat((box[i], x[i, j + 5].unsqueeze(1),
                           j.float().unsqueeze(1)), 1)
        else:
            conf, j = x[:, 5:].max(1)
            x = torch.cat(
                (box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)[
                conf > conf_thres]

        if classes:
            x = x[(j.view(-1, 1) == torch.tensor(classes,
                                                 device=j.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue

        c = x[:, 5] * 0 if agnostic else x[:, 5]
        boxes, scores = x[:, :4].clone() + c.view(-1, 1) * max_wh, x[:, 4]
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if merge and (1 < n < 3E3):
            try:
                iou = box_iou(boxes[i], boxes) > iou_thres
                weights = iou * scores[None]
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            except BaseException:
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break

    return output
