import cv2
import numpy as np
import glob
import random
import os

# Images path
# images_path = glob.glob(r"D:\Other\Faks\Slikovna Biometrija - SB\s2\data\ears\test\*.png")


class MySuperDetector:
    # Load Yolo
    # net = cv2.dnn.readNet("yolov3_training_800.weights", "yolov3_testing.cfg")

    # YOLOv1
    # net = cv2.dnn.readNet(
    #     os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yolov1/yolo_training_last.weights'),
    #     os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yolov1/yolo_testing.cfg')
    # )

    # YOLOv2
    # net = cv2.dnn.readNet(
    #     os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yolov2/yolov2_training_last.weights'),
    #     os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yolov2/yolov2_testing.cfg')
    # )

    # YOLOv3 800
    net = cv2.dnn.readNet(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yolov3/800/yolov3_training_800.weights'),
        os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yolov3/yolov3_testing.cfg')
    )

    # YOLOv3 1000
    # net = cv2.dnn.readNet(
    #     os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yolov3/1000/yolov3_training_regular_1000.weights'),
    #     os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yolov3/yolov3_testing.cfg')
    # )

    # YOLOv3 1200
    # net = cv2.dnn.readNet(
    #     os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yolov3/1200/yolov3_training_regular_last.weights'),
    #     os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yolov3/yolov3_testing.cfg')
    # )

    # YOLOv3 Resolution
    # net = cv2.dnn.readNet(
    #     os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yolov3/resolution/yolov3_training_resolution_last.weights'),
    #     os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yolov3/yolov3_testing.cfg')
    # )

    # YOLOv3 Brightness 1000
    # net = cv2.dnn.readNet(
    #     os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yolov3/brightness/yolov3_training_brightness_1000.weights'),
    #     os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yolov3/yolov3_testing.cfg')
    # )

    # YOLOv3 Brightness 1600
    # net = cv2.dnn.readNet(
    #     os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yolov3/brightness/yolov3_training_brightness_last.weights'),
    #     os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yolov3/yolov3_testing.cfg')
    # )

    # YOLOv3 Tiny Occlusion
    # net = cv2.dnn.readNetFromDarknet(
    #     os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yolov3/occlusion/yolov3-tiny_occlusion_track_training.cfg'),
    #     os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yolov3/occlusion/yolov3-tiny_occlusion_track_training_last.weights')
    # )

    # # YOLOv4 Tiny
    # net = cv2.dnn.readNet(
    #     os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yolov4/yolov4-tiny_training_last.weights'),
    #     os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yolov4/yolov4-tiny_testing.cfg')
    # )



    def detect(self, img):
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        # Loading image
        # img = cv2.imread(img_path)
        # img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        self.net.setInput(blob)
        outs = self.net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    # Object detected
                    print(class_id)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        yolo_boxes = []
        yolo_confidences = []
        for i in range(len(boxes)):
            if i in indexes:
                yolo_boxes.append(boxes[i])
                yolo_confidences.append(confidences[i])
        return yolo_boxes, yolo_confidences


if __name__ == '__main__':
    # fname = sys.argv[1]
    im_list = sorted(glob.glob('../../data/ears/test' + '/*.png', recursive=True))
    fname = im_list[0]
    img = cv2.imread(fname)
    detector = MySuperDetector()
    detected_loc = detector.detect(img)
    for x, y, w, h in detected_loc:
        cv2.rectangle(img, (x,y), (x+w, y+h), (128, 255, 0), 4)
    cv2.imwrite(fname + '.detected.jpg', img)