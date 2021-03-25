import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import threading
import multiprocessing
from time import sleep
import pytesseract
from PIL import Image
import uuid

options = {
    'model': 'cfg/yolov2-tiny.cfg',
    'load': 'bin/yolov2-tiny_3000.weights',
    'threshold': 0.1,
    'gpu': 0.5
}

pytesseract.pytesseract.tesseract_cmd='C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


tfnet = TFNet(options)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

#Save Image to local directory
def saveImage(origialImage, state):
    print("Save Helmet not Wear Image")
    filename = str(uuid.uuid4())
    imgName = "PenaultyImages/"+filename + ".jpg"
    print("Saving Image "+imgName)
    cv2.imwrite(imgName,origialImage)
    gray = cv2.cvtColor(origialImage, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cv2.imwrite('LatestFine.jpg',thresh)
    data = pytesseract.image_to_string(thresh, lang='eng',config='--psm 6')
    print("Send Fine to Vehical : "+data)


while True:
    stime = time.time()
    ret, img = capture.read()
    if ret:
        results = tfnet.return_predict(img)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']
            text = '{}: {:.0f}%'.format(label, confidence * 100)
            print(text)
            print("Helmet Status "+label)
            #Check Helmet Wearing status and save image to local
            if (label == "No_Helmet"):
                saveImage(img, "Ok")

            frame = cv2.rectangle(img, tl, br, color, 5)
            frame = cv2.putText(img, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

        cv2.imshow('frame', frame)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()