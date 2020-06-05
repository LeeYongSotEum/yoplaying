import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

img_paths = {}
descs = {}

for i in range(150):
    name = 'WonBin_'+ str(i+1)
    path = 'image/WonBin/WonBin ('+ str(i+1) +').jpg'
    img_paths.update({name:path})
    descs.update({name:None})


detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

def find_faces(img):
    dets = detector(img, 1)

    if len(dets) == 0:
        return 0, 0, 0

    if len(dets) > 2:
        return 2, 2, 2

    rects, shapes = [], []
    shapes_np = np.zeros((len(dets), 68, 2), dtype=np.int)

    for k, d in enumerate(dets):
        rect = ((d.left(), d.top()),(d.right(), d.bottom()))
        rects.append(rect)

        shape = sp(img, d)

        for i in range(0, 68):
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        shapes.append(shape)

    return  rects, shapes, shapes_np

def encode_faces(img, shapes):
    face_descriptors = []

    for shape in shapes:
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        face_descriptors.append(np.array(face_descriptor))

    return np.array(face_descriptors)



for name, img_path in img_paths.items():
    # print(name)
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    _, img_shapes, _ = find_faces(img_rgb)
    if img_shapes == 0:
        print("못찾은 이미지 :", name)
    if img_shapes == 2:
        print('얼굴이 2개 이상 :', name)
        # pass
    else:
        # descs[name] = encode_faces(img_rgb, img_shapes)[0]
        pass
    # print(descs[name])

# np.save('image/descs.npy', descs)
# print(descs)
print('Done')