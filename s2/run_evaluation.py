import cv2
import numpy as np
import glob
import os
from pathlib import Path
import json
from preprocessing.preprocess import Preprocess
from metrics.evaluation import Evaluation
from mean_average_precision import MetricBuilder

class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config.json') as config_file:
            config = json.load(config_file)

        self.images_path = config['images_path']
        self.annotations_path = config['annotations_path']

    def get_annotations(self, annot_name):
            with open(annot_name) as f:
                lines = f.readlines()
                annot = []
                for line in lines:
                    l_arr = line.split(" ")[1:5]
                    l_arr = [int(i) for i in l_arr]
                    annot.append(l_arr)
            return annot

    def run_evaluation(self):

        im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
        iou_arr = []
        ground_trouth = []
        prediction = []
        preprocess = Preprocess()
        eval = Evaluation()
        
        # Change the following detector and/or add your detectors below
        import detectors.cascade_detector.detector as cascade_detector
        # import detectors.your_super_detector.detector as super_detector
        import detectors.your_super_detector.mySuperDetector as super_detector
        cascade_detector = cascade_detector.Detector()
        super_detector = super_detector.MySuperDetector()
        

        # # Preproccesing - run only once, since it overwrites images


        # data_size = len(im_list)
        #
        # channels = 3
        # IMG_SIZE_W = 480
        # IMG_SIZE_H = 360
        #
        # dataset = np.ndarray(shape=(data_size, IMG_SIZE_H, IMG_SIZE_W, channels),
        #                      dtype=np.float32)
        # i = 0
        # for img in im_list:
        #     image = cv2.imread(img)
        #     image = cv2.resize(image, (IMG_SIZE_W, IMG_SIZE_H))
        #
        #     dataset[i] = image
        #     i += 1
        #
        # for i, im in enumerate(dataset):
        #     imgg = im.astype('uint8')
        #
        #     # Brightness correction
        #     image_bright, alpha, beta = preprocess.automatic_brightness_and_contrast(imgg)
        #
        #     # edge enhancment
        #     kernel = np.array([[0, -1, 0],
        #                        [-1, 5, -1],
        #                        [0, -1, 0]])
        #     image_sharp = cv2.filter2D(src=image_bright, ddepth=-1, kernel=kernel)
        #
        #     # Histogram equalization
        #     intensity_img = cv2.cvtColor(image_sharp, cv2.COLOR_BGR2YCrCb)
        #     intensity_img[:, :, 0] = cv2.equalizeHist(intensity_img[:, :, 0])
        #
        #     dataset[i] = cv2.cvtColor(intensity_img, cv2.COLOR_YCrCb2BGR)
        #
        # # Mean-centred image subtraction
        # # img_dset = preprocess.mean_centred_sub(dataset)
        # # mean-centred image
        # mean = dataset.mean(axis=(0, 1, 2))
        #
        # dataset[..., 0] -= mean[0]
        # dataset[..., 1] -= mean[1]
        # dataset[..., 2] -= mean[2]

        # for i, im in enumerate(dataset):
        # #     dataset[i] = dataset[i]
        #     im_name = im_list[i]
        #     cv2.imwrite(im_name, im)
            # if(i==1):
            #     break

        # detection, evaluation
        # for img_iter, img in enumerate(dataset):
        for im_name in im_list:
            img = cv2.imread(im_name)

            #ablation study - resolution
            # IMG_SIZE_W = int(480/2)
            # IMG_SIZE_H = int(360/2)
            # img = cv2.resize(img, (IMG_SIZE_W, IMG_SIZE_H))

            # ablation study - brightness
            # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # value = 42
            # hsv[:, :, 2] += value
            # img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


            # Run the detector. It runs a list of all the detected bounding-boxes. In segmentor you only get a mask matrices, but use the iou_compute in the same way.
            # prediction_list = cascade_detector.detect(img)
            prediction_list, confidences = super_detector.detect(img)

            # Read annotations:
            # im_name = im_list[img_iter]
            annot_name = os.path.join(self.annotations_path, Path(os.path.basename(im_name)).stem) + '.txt'
            annot_list = self.get_annotations(annot_name)

            # [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
            for box in annot_list:
                ground_trouth.append([box[0], box[1], box[0] + box[2], box[1] + box[3], 0, 0, 0])

            # [xmin, ymin, xmax, ymax, class_id, confidence]
            for ind, pred_box in enumerate(prediction_list):
                prediction.append([pred_box[0], pred_box[1], pred_box[0] + pred_box[2], pred_box[1] + pred_box[3], 0, confidences[ind]])
            # Only for detection:
            p, gt = eval.prepare_for_detection(prediction_list, annot_list)
            
            iou = eval.iou_compute(p, gt)
            iou_arr.append(iou)

        miou = np.average(iou_arr)
        print("\n")
        print("Average IOU:", f"{miou:.2%}")
        print("\n")
        prediction_np = np.array(prediction)
        ground_trouth_np = np.array(ground_trouth)
        metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)
        metric_fn.add(prediction_np, ground_trouth_np)
        # compute PASCAL VOC metric
        print(f"VOC PASCAL mAP: {metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']}")

        # compute PASCAL VOC metric at the all points
        print(f"VOC PASCAL mAP in all points: {metric_fn.value(iou_thresholds=0.5)['mAP']}")

        # compute metric COCO metric
        print(f"COCO mAP: {metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}")


if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()