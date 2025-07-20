#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import argparse
import cv2
import numpy as np
import onnxruntime as ort

from quark.onnx import get_library_path
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml


class Yolov8:

    def __init__(self, onnx_model, input_image, confidence_thres, iou_thres):
        self.onnx_model = onnx_model
        self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        self.classes = yaml_load(check_yaml('coco128.yaml'))['names']

        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, img, box, score, class_id):
        x1, y1, w, h = box

        color = self.color_palette[class_id]

        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        label = f'{self.classes[class_id]}: {score:.2f}'

        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                      cv2.FILLED)

        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self):
        self.img = cv2.imread(self.input_image)

        self.img_height, self.img_width = self.img.shape[:2]

        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (self.input_width, self.input_height))

        image_data = np.array(img) / 255.0

        image_data = np.transpose(image_data, (2, 0, 1))

        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        return image_data

    def postprocess(self, input_image, output):
        outputs = np.transpose(np.squeeze(output[0]))

        rows = outputs.shape[0]

        boxes = []
        scores = []
        class_ids = []

        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        for i in range(rows):
            classes_scores = outputs[i][4:]

            max_score = np.amax(classes_scores)

            if max_score >= self.confidence_thres:
                class_id = np.argmax(classes_scores)

                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            self.draw_detections(input_image, box, score, class_id)

        return input_image

    def infer(self):
        sess_options = ort.SessionOptions()
        sess_options.register_custom_ops_library(get_library_path())
        session = ort.InferenceSession(self.onnx_model, sess_options, providers=['CPUExecutionProvider'])

        model_inputs = session.get_inputs()

        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        img_data = self.preprocess()

        outputs = session.run(None, {model_inputs[0].name: img_data})

        return self.postprocess(self.img, outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model_path', type=str, help='Input your ONNX model.')
    parser.add_argument('--input_image', type=str, help='Path to input image.')
    parser.add_argument('--output_image', type=str, help='Path to output image.')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    args = parser.parse_args()

    detection = Yolov8(args.input_model_path, args.input_image, args.conf_thres, args.iou_thres)
    output_image = detection.infer()
    cv2.imwrite(args.output_image, output_image)
