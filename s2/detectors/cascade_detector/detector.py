import cv2, sys, os
import glob

class Detector:
	# This example of a detector detects faces. However, you have annotations for ears!

	cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'lbpcascade_frontalface.xml'))
	# cascade = cv2.CascadeClassifier("cascades/haarcascade_mcs_leftear.xml")
	# cascade = cv2.CascadeClassifier("cascades/haarcascade_mcs_rightear.xml")

	def detect(self, img):
		det_list = self.cascade.detectMultiScale(img, 1.05, 1)
		return det_list

if __name__ == '__main__':
	# fname = sys.argv[1]
	im_list = sorted(glob.glob('../../data/ears/test' + '/*.png', recursive=True))
	fname = im_list[0]
	img = cv2.imread(fname)
	detector = Detector()
	detected_loc = detector.detect(img)
	for x, y, w, h in detected_loc:
		cv2.rectangle(img, (x,y), (x+w, y+h), (128, 255, 0), 4)
	cv2.imwrite(fname + '.detected.jpg', img)