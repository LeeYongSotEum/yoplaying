import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

img = cv2.imread('image/SungSiKyung/SungSiKyung (1).jpg')

faces = detector(img)
if len(faces) > 1:
    print('한명의 사진만 올려주세요')
elif len(faces) == 0:
    print('얼굴을 찾을 수 없습니다 되도록이면 정면사진을 올려주세요')
else:
    face = faces[0]
    img = cv2.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(), face.bottom()), color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow('img', img)
    cv2.waitKey(0)
