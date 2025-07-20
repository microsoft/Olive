#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
import glob
import math
from PIL import ExifTags, Image
import shutil
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils import box_iou, plot_one_box, clip_coords, scale_coords, xywh2xyxy, letterbox, non_max_suppression
import onnxruntime
import cv2
import matplotlib.pyplot as plt

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

help_url = 'https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data'
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']
vid_formats = ['.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']



def create_folder(path='./new_folder'):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def exif_size(img):
    s = img.size
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:
            s = (s[1], s[0])
        elif rotation == 8:
            s = (s[1], s[0])
    except BaseException:
        pass

    return s


def ap_per_class(tp, conf, pred_cls, target_cls):

    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes = np.unique(target_cls)

    pr_score = 0.1
    s = [unique_classes.shape[0], tp.shape[1]]
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()
        n_p = i.sum()

        if n_p == 0 or n_gt == 0:
            continue
        else:
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            recall = tpc / (n_gt + 1e-16)
            r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])

            precision = tpc / (tpc + fpc)
            p[ci] = np.interp(-pr_score, -conf[i],
                              precision[:, 0])

            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def plot_images(
        images,
        targets,
        paths=None,
        fname='images.jpg',
        names=None,
        max_size=640,
        max_subplots=16):
    tl = 3
    tf = max(tl - 1, 1)
    if os.path.isfile(fname):
        return None

    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()

    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    if np.max(images[0]) <= 1:
        images *= 255

    bs, _, h, w = images.shape
    bs = min(bs, max_subplots)
    ns = np.ceil(bs ** 0.5)

    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)

    prop_cycle = plt.rcParams['axes.prop_cycle']

    def hex2rgb(h):
        return tuple(
            int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    color_lut = [hex2rgb(h) for h in prop_cycle.by_key()['color']]

    for i, img in enumerate(images):
        if i == max_subplots:
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            classes = image_targets[:, 1].astype('int')
            gt = image_targets.shape[1] == 6
            conf = None if gt else image_targets[:, 6]

            boxes[[0, 2]] *= w
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] *= h
            boxes[[1, 3]] += block_y
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = color_lut[cls % len(color_lut)]
                cls = names[cls] if names else cls
                if gt or conf[j] > 0.3:
                    label = '%s' % cls if gt else '%s %.1f' % (cls, conf[j])
                    plot_one_box(box, mosaic, label=label,
                                 color=color, line_thickness=tl)

        if paths is not None:
            label = os.path.basename(paths[i])[:40]
            t_size = cv2.getTextSize(
                label, 0, fontScale=tl / 3, thickness=tf)[0]
            cv2.putText(mosaic, label, (block_x +
                                        5, block_y +
                                        t_size[1] +
                                        5), 0, tl /
                        3, [220, 220, 220], thickness=tf, lineType=cv2.LINE_AA)

        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w,
                                                   block_y + h), (255, 255, 255), thickness=3)

    if fname is not None:
        mosaic = cv2.resize(mosaic,
                            (int(ns * w * 0.5),
                             int(ns * h * 0.5)),
                            interpolation=cv2.INTER_AREA)
        cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))

    return mosaic


def random_affine(img, targets=(), degrees=10, translate=.1,
                  scale=.1, shear=10, border=0):
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a,
                                    center=(img.shape[1] / 2,
                                            img.shape[0] / 2),
                                    scale=s)

    T = np.eye(3)
    T[0, 2] = (random.uniform(-translate, translate) *
               img.shape[0] + border)
    T[1, 2] = (random.uniform(-translate, translate) *
               img.shape[1] + border)

    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) *
                       math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) *
                       math.pi / 180)

    M = S @ T @ R
    if (border != 0) or (M != np.eye(3)).any():
        img = cv2.warpAffine(img, M[:2], dsize=(width, height),
                             flags=cv2.INTER_LINEAR,
                             borderValue=(114, 114, 114))

    n = len(targets)
    if n:
        xy = np.ones((n * 4, 3))
        xy[:, :2] = (targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].
                     reshape(n * 4, 2))
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1),
                             y.max(1))).reshape(4, n).T

        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = ((targets[:, 3] - targets[:, 1]) *
                 (targets[:, 4] - targets[:, 2]))
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16)
                                 > 0.2) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets


def output_to_target(output, width, height):
    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()

    targets = []
    for i, o in enumerate(output):
        if o is not None:
            for pred in o:
                box = pred[:4]
                w = (box[2] - box[0]) / width
                h = (box[3] - box[1]) / height
                x = box[0] / width + w / 2
                y = box[1] / height + h / 2
                conf = pred[4]
                cls = int(pred[5])

                targets.append([i, cls, x, y, w, h, conf])

    return np.array(targets)


def xyxy2xywh(x):
    y = torch.zeros_like(x) if isinstance(
        x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def coco80_to_coco91_class():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
         21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
         41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
         59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
         80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def check_file(file):
    if os.path.isfile(file):
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)
        assert len(files), 'File Not Found: %s' % file
        return files[0]


def load_classes(path):
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))


def load_image(self, index):
    img = self.imgs[index]
    if img is None:
        path = self.img_files[index]
        img = cv2.imread(path)
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]
        r = self.img_size / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 and not self.augment \
                else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                             interpolation=interp)
        return img, (h0, w0), img.shape[:2]
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]


def load_mosaic(self, index):
    labels4 = []
    s = self.img_size
    xc, yc = [int(random.uniform(s * 0.5, s * 1.5))
              for _ in range(2)]
    indices = [index] + [random.randint(0, len(self.labels) - 1)
                         for _ in range(3)]
    for i, index in enumerate(indices):
        img, _, (h, w) = load_image(self, index)

        if i == 0:
            img4 = np.full((s * 2, s * 2, img.shape[2]),
                           114, dtype=np.uint8)
            x1a, y1a, x2a, y2a = (max(xc - w, 0),
                                  max(yc - h, 0), xc, yc)
            x1b, y1b, x2b, y2b = (w - (x2a - x1a), h -
                                  (y2a - y1a), w, h)
        elif i == 1:
            x1a, y1a, x2a, y2a = (xc, max(yc - h, 0),
                                  min(xc + w, s * 2), yc)
            x1b, y1b, x2b, y2b = (0, h - (y2a - y1a),
                                  min(w, x2a - x1a), h)
        elif i == 2:
            x1a, y1a, x2a, y2a = (max(xc - w, 0), yc,
                                  xc, min(s * 2, yc + h))
            x1b, y1b, x2b, y2b = (w - (x2a - x1a), 0,
                                  max(xc, w), min(y2a - y1a, h))
        elif i == 3:
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w,
                                             s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = (0, 0,
                                  min(w, x2a - x1a), min(y2a - y1a, h))

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        padw = x1a - x1b
        padh = y1a - y1b

        x = self.labels[index]
        labels = x.copy()
        if x.size > 0:
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
        labels4.append(labels)

    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])

    img4, labels4 = random_affine(img4, labels4,
                                  degrees=self.hyp['degrees'],
                                  translate=self.hyp['translate'],
                                  scale=self.hyp['scale'],
                                  shear=self.hyp['shear'],
                                  border=-s // 2)

    return img4, labels4


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    method = 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)
        ap = np.trapz(np.interp(x, mrec, mpre), x)
    else:

        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = (np.random.uniform(-1, 1, 3) *
         [hgain, sgain, vgain] + 1)
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue),
                         cv2.LUT(sat, lut_sat),
                         cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)



class LoadImagesAndLabels(Dataset):
    def __init__(
            self,
            path,
            img_size=416,
            batch_size=16,
            augment=False,
            hyp=None,
            rect=False,
            image_weights=False,
            cache_images=False,
            single_cls=False,
            pad=0.0):
        try:
            path = str(Path(path))
            parent = str(Path(path).parent) + os.sep
            if os.path.isfile(path):
                with open(path, 'r') as f:
                    f = f.read().splitlines()

                    f = [
                        x.replace(
                            './',
                            parent) if x.startswith('./') else x for x in f]
            elif os.path.isdir(path):
                f = glob.iglob(path + os.sep + '*.*')
            else:
                raise Exception('%s does not exist' % path)
            self.img_files = [x.replace(
                '/', os.sep) for x in f if
                os.path.splitext(x)[-1].lower() in img_formats]

        except BaseException:
            raise Exception(
                'Error loading data from %s. See %s' %
                (path, help_url))

        n = len(self.img_files)
        assert n > 0, 'No images found in %s. See %s' % (path, help_url)
        bi = np.floor(np.arange(n) / batch_size).astype(int)
        nb = bi[-1] + 1

        self.n = n
        self.batch = bi
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect


        self.label_files = [x.replace('images', 'labels').replace(
            os.path.splitext(x)[-1], '.txt') for x in self.img_files]

        sp = path.replace('.txt', '') + '.shapes'
        try:
            with open(sp, 'r') as f:  #
                s = [x.split() for x in f.read().splitlines()]
                assert len(s) == n, 'Shapefile out of sync'
        except BaseException:
            s = [exif_size(Image.open(f)) for f in tqdm(
                self.img_files,
                desc='Reading image shapes')]
            np.savetxt(sp, s, fmt='%g')

        self.shapes = np.array(s, dtype=np.float64)

        if self.rect:
            s = self.shapes
            ar = s[:, 1] / s[:, 0]
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.shapes = s[irect]
            ar = ar[irect]

            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(
                np.array(shapes) * img_size / 32. + pad).astype(int) * 32

        self.imgs = [None] * n
        self.labels = [np.zeros((0, 5), dtype=np.float32)] * n
        create_datasubset, extract_bounding_boxes, labels_loaded = \
            False, False, False
        nm, nf, ne, ns, nd = 0, 0, 0, 0, 0
        np_labels_path = str(Path(self.label_files[0]).parent) + '.npy'
        if os.path.isfile(np_labels_path):
            s = np_labels_path

            print(np_labels_path)

            x = np.load(np_labels_path, allow_pickle=True)
            if len(x) == n:
                self.labels = x
                labels_loaded = True
        else:
            s = path.replace('images', 'labels')

        pbar = tqdm(self.label_files)
        for i, file in enumerate(pbar):
            if labels_loaded:
                l = self.labels[i]
            else:
                try:
                    with open(file, 'r') as f:
                        l = np.array(
                            [x.split() for x in f.read().splitlines()],
                            dtype=np.float32)
                except BaseException:
                    nm += 1
                    continue

            if l.shape[0]:
                assert l.shape[1] == 5, '> 5 label columns: %s' % file
                assert (l >= 0).all(), 'negative labels: %s' % file
                assert (l[:, 1:] <= 1).all(
                ), 'non-normalized or out of bounds coordinate labels: %s' % file
                if np.unique(
                        l, axis=0).shape[0] < l.shape[0]:
                    nd += 1
                if single_cls:
                    l[:, 0] = 0
                self.labels[i] = l
                nf += 1

                if create_datasubset and ns < 1E4:
                    if ns == 0:
                        create_folder(path='./datasubset')
                        os.makedirs('./datasubset/images')
                    exclude_classes = 43
                    if exclude_classes not in l[:, 0]:
                        ns += 1
                        with open('./datasubset/images.txt', 'a') as f:
                            f.write(self.img_files[i] + '\n')

                if extract_bounding_boxes:
                    p = Path(self.img_files[i])
                    img = cv2.imread(str(p))
                    h, w = img.shape[:2]
                    for j, x in enumerate(l):
                        f = '%s%sclassifier%s%g_%g_%s' % (
                            p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                        if not os.path.exists(Path(f).parent):
                            os.makedirs(Path(f).parent)

                        b = x[1:] * [w, h, w, h]
                        b[2:] = b[2:].max()
                        b[2:] = b[2:] * 1.3 + 30
                        b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(int)

                        b[[0, 2]] = np.clip(b[[0, 2]], 0, w)
                        b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                        assert cv2.imwrite(
                            f, img[b[1]:b[3], b[0]:b[2]]), \
                            'Failure extracting classifier boxes'
            else:
                ne += 1

            pbar.desc = 'Caching labels %s (%g found, %g missing, %g empty,\
             %g duplicate, for %g images)' % (
                s, nf, nm, ne, nd, n)
        assert nf > 0 or n == 20288, 'No labels found in %s. See %s' % (
            os.path.dirname(file) + os.sep, help_url)
        if not labels_loaded and n > 1000:
            print(
                'Saving labels to %s for faster future loading' %
                np_labels_path)

        if cache_images:
            gb = 0
            pbar = tqdm(range(len(self.img_files)), desc='Caching images')
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            for i in pbar:
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = load_image(
                    self, i)
                gb += self.imgs[i].nbytes
                pbar.desc = 'Caching images (%.1fGB)' % (gb / 1E9)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        if self.image_weights:
            index = self.indices[index]

        hyp = self.hyp
        if self.mosaic:
            img, labels = load_mosaic(self, index)
            shapes = None

        else:
            img, (h0, w0), (h, w) = load_image(self, index)

            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)

            labels = []
            x = self.labels[index]
            if x.size > 0:
                labels = x.copy()
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        if self.augment:
            if not self.mosaic:
                img, labels = random_affine(img, labels,
                                            degrees=hyp['degrees'],
                                            translate=hyp['translate'],
                                            scale=hyp['scale'],
                                            shear=hyp['shear'])

            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

        nL = len(labels)
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            labels[:, [2, 4]] /= img.shape[0]
            labels[:, [1, 3]] /= img.shape[1]

        if self.augment:
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes


def parse_data_cfg(path):
    if not os.path.exists(path) and os.path.exists(
            'data' + os.sep + path):
        path = 'data' + os.sep + path

    with open(path, 'r') as f:
        lines = f.readlines()

    options = dict()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, val = line.split('=')
        options[key.strip()] = val.strip()

    return options


def create_grids(ng=(13, 13), device='cpu'):
    nx, ny = ng
    ng = torch.tensor(ng, dtype=torch.float)

    yv, xv = torch.meshgrid(
        [torch.arange(ny, device=device), torch.arange(nx, device=device)])
    grid = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    return grid


def post_process(x):
    stride = [32, 16, 8]
    anchors = [[10, 13, 16, 30, 33, 23],
               [30, 61, 62, 45, 59, 119],
               [116, 90, 156, 198, 373, 326]]
    temp = [13, 26, 52]

    res = []
    for i in range(3):
        out = torch.from_numpy(x[i]) if not torch.is_tensor(x[i]) else x[i]

        bs, _, ny, nx = out.shape

        anchor = torch.Tensor(anchors[2 - i]).reshape(3, 2)
        anchor_vec = anchor / stride[i]
        anchor_wh = anchor_vec.view(1, 3, 1, 1, 2)

        grid = create_grids((nx, ny))

        out = out.view(
            bs, 3, 85, temp[i], temp[i]).permute(
            0, 1, 3, 4, 2).contiguous()

        io = out.clone()

        io[..., :2] = torch.sigmoid(io[..., :2]) + grid
        io[..., 2:4] = torch.exp(io[..., 2:4]) * anchor_wh
        io[..., :4] *= stride[i]
        torch.sigmoid_(io[..., 4:])

        res.append(io.view(bs, -1, 85))
    return torch.cat(res, 1), x


def test(data,
         batch_size=32,
         imgsz=416,
         conf_thres=0.001,
         iou_thres=0.6,
         save_json=False,
         single_cls=False,
         augment=False,
         model=None,
         dataloader=None,
         multi_label=True,
         names='data/coco.names',
         onnx_runtime=True,
         onnx_weights="yolov3-8",
         ipu=False,
         provider_config='vaip_config.json'):

    device = torch.device('cpu')
    verbose = False
    if isinstance(onnx_weights, list):
        onnx_weights = onnx_weights[0]

    if ipu:
        providers = ["VitisAIExecutionProvider"]
        provider_options = [{"config_file": provider_config}]
    else:
        providers = ['CPUExecutionProvider', 'ROCMExecutionProvider', 'CUDAExecutionProvider']
        provider_options = None

    onnx_model = onnxruntime.InferenceSession(
        onnx_weights,
        providers=providers,
        provider_options=provider_options)

    data = parse_data_cfg(data)
    nc = 1 if single_cls else int(data['classes'])
    path = data['valid']
    names = load_classes(data['names'])
    iouv = torch.linspace(0.5, 0.95, 10).to(
        device)
    iouv = iouv[0].view(1)
    niou = iouv.numel()

    if dataloader is None:
        dataset = LoadImagesAndLabels(
            path,
            imgsz,
            batch_size,
            rect=False,
            single_cls=single_cls,
            pad=0.5)
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=min([os.cpu_count(),
                                                 batch_size if
                                                 batch_size > 1 else 0,
                                                 8]),
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)

    seen = 0

    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R',
                                 'mAP@0.5', 'F1')
    p, r, f1, mp, mr, map, mf1, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []

    for batch_i, (imgs, targets, paths, shapes) in enumerate(
            tqdm(dataloader, desc=s)):
        imgs = imgs.to(device).float() / 255.0
        targets = targets.to(device)
        nb, _, height, width = imgs.shape
        whwh = torch.Tensor([width, height, width, height]).to(device)

        if onnx_runtime:
            save_sample = False
            if save_sample:
                one_sample = np.transpose(imgs.cpu().numpy(), (0, 2, 3, 1))
                one_sample_path = f"./cali_dataset/sample_{batch_i}.npy"
                np.save(one_sample_path, one_sample)


            outputs = onnx_model.run(
                None, {onnx_model.get_inputs()[0].name: np.transpose(imgs.cpu().numpy(), (0, 2, 3, 1))})
            outputs = [np.transpose(out, (0, 3, 1, 2)) for out in outputs]

            outputs = [torch.tensor(item).to(device) for item in outputs]
            inf_out, train_out = post_process(outputs)

        else:
            with torch.no_grad():
                t = time_synchronized()

                inf_out, train_out = model(imgs, augment=augment)
                t0 += time_synchronized() - t

        t = time_synchronized()
        output = non_max_suppression(
            inf_out,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            multi_label=multi_label)
        t1 += time_synchronized() - t

        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []
            seen += 1

            if pred is None:
                if nl:
                    stats.append(
                        (torch.zeros(
                            0,
                            niou,
                            dtype=torch.bool),
                         torch.Tensor(),
                         torch.Tensor(),
                         tcls))
                continue

            clip_coords(pred, (height, width))

            if save_json:
                image_id = int(Path(paths[si]).stem.split('_')[-1])
                box = pred[:, :4].clone()
                scale_coords(imgs[si].shape[1:], box, shapes[si][0], shapes[si][1])
                box = xyxy2xywh(box)
                box[:, :2] -= box[:, 2:] / 2
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])],
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            correct = torch.zeros(
                pred.shape[0],
                niou,
                dtype=torch.bool,
                device=device)
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]

                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(
                    ).view(-1)
                    pi = (cls == pred[:, 5]).nonzero(
                    ).view(-1)


                    if pi.shape[0]:

                        ious, i = box_iou(pred[pi, :4], tbox[ti].cpu()).max(
                            1)

                        for j in (ious > iouv[0].cpu()).nonzero():
                            d = ti[i[j]]
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv.cpu()
                                if len(
                                        detected) == nl:
                                    break

            stats.append(
                (correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        if batch_i < 1:
            f = 'test_batch%g_gt.jpg' % batch_i
            plot_images(imgs, targets, paths=paths, names=names,
                        fname=f)
            f = 'test_batch%g_pred.jpg' % batch_i
            plot_images(imgs, output_to_target(output, width, height),
                        paths=paths, names=names, fname=f)

    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        if niou > 1:
            p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(
                1), ap[:, 0]
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64),
                         minlength=nc)
    else:
        nt = torch.zeros(1)

    pf = '%20s' + '%10.3g' * 6
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    if verbose or save_json:
        t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + \
            (imgsz, imgsz, batch_size)
        print(
            'Speed: %.1f/%.1f/%.1f ms \
            inference/NMS/total per %gx%g image at batch-size %g' % t)

    if save_json and map and len(jdict):
        print('\nCOCO mAP with pycocotools...')
        imgIds = [int(Path(x).stem.split('_')[-1])
                  for x in dataloader.dataset.img_files]
        with open('results.json', 'w') as file:
            json.dump(jdict, file)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            cocoGt = COCO(
                glob.glob('coco/annotations/instances_val*.json')[0])
            cocoDt = cocoGt.loadRes('results.json')
            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
        except BaseException:
            print(
                'WARNING: pycocotools must be installed with \
                numpy==1.17 to run correctly. '
                'See https://github.com/cocodataset/cocoapi/issues/356')

    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map, mf1, *(loss.cpu() / len(dataloader)).tolist()), maps


def evaluator(onnx_path, data='./coco2017.data', batch_size=1, img_size=416, conf_thres=0.001, iou_thres=0.5,
              save_json=False, single_cls=False, augment=False, ipu=False, ):
    save_json = save_json or any([x in data for x in ['coco.data', 'coco2014.data', 'coco2017.data']])
    data = check_file(data)

    result = test(data, batch_size, img_size, conf_thres, iou_thres, save_json, single_cls, augment, names='data/coco.names', onnx_weights=onnx_path, ipu=ipu, onnx_runtime=True)

    return result[0][2]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Test onnx model performance on COCO dataset')
    parser.add_argument(
        '--data',
        type=str,
        default='coco2017.data',
        help='Path of *.data')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Size of each image batch')
    parser.add_argument(
        '--img-size',
        type=int,
        default=416,
        help='Inference size (pixels)')
    parser.add_argument(
        '--conf-thres',
        type=float,
        default=0.001,
        help='Object confidence threshold')
    parser.add_argument(
        '--iou-thres',
        type=float,
        default=0.5,
        help='IOU threshold for NMS')
    parser.add_argument(
        '--save-json',
        action='store_true',
        help='Save a COCOapi-compatible JSON results file')
    parser.add_argument(
        '--device',
        default='',
        help='Device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument(
        '--augment',
        action='store_true',
        help='Augmented inference')
    parser.add_argument('--sync_bn', action='store_true')
    parser.add_argument('--print_model', action='store_true')
    parser.add_argument('--test_rect', action='store_true')

    parser.add_argument(
        '--onnx_runtime',
        action='store_true',
        help='Use onnx runtime')
    parser.add_argument(
        '--onnx_weights',
        default='yolov3-8.onnx',
        nargs='+',
        type=str,
        help='Path of onnx weights')
    parser.add_argument(
        '--single-cls',
        action='store_true',
        help='Run as single-class dataset')
    parser.add_argument(
        "--ipu",
        action="store_true",
        help="Use IPU for inference")
    parser.add_argument(
        "--provider_config",
        type=str,
        default="vaip_config.json",
        help="Path of the config file for seting provider_options")

    opt = parser.parse_args()
    opt.save_json = opt.save_json or any(
        [x in opt.data for x in ['coco.data',
                                 'coco2014.data', 'coco2017.data']])
    opt.data = check_file(opt.data)
    print(opt)


    test(opt.data,
         opt.batch_size,
         opt.img_size,
         opt.conf_thres,
         opt.iou_thres,
         opt.save_json,
         opt.single_cls,
         opt.augment,
         names='data/coco.names',
         onnx_weights=opt.onnx_weights,
         ipu=opt.ipu,
         provider_config=opt.provider_config
         )
