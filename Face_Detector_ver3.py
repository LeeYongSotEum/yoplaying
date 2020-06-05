import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

def find_faces(img):
    dets = detector(img, 1)

    if len(dets) == 0:
        return np.empty(0), np.empty(0), np.empty(0)

    if len(dets) > 2:
        return np.empty(0), np.empty(0), np.empty(0)

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

descs = np.load('image/descs.npy', allow_pickle=True).item()

img_bgr = cv2.imread('image/2.jpg')
print(img_bgr)
print(type(img_bgr))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
rects, shapes, _ = find_faces(img_rgb)
descriptors = encode_faces(img_rgb, shapes)


fig, ax = plt.subplots(1, figsize=(5, 5))
ax.imshow(img_rgb)

for i, desc in enumerate(descriptors):
    dist_dic = {}
    for name, saved_desc in descs.items():
        dist = np.linalg.norm([desc]-saved_desc, axis=1)
        dist = 1-dist
        dist_dic.update({name:dist})

    res = sorted(dist_dic.items(), key=(lambda x:x[1]), reverse=True)
    print(res)

    result = []
    # result_per = []
    for idx in range(100):
        temp = res[idx][0].split('_')[0]
        result.append(temp)
        temp_per = res[idx][1]
    #     if temp_per > 0.55:
    #         result_per.append(temp_per)
    #
    # for idx in range(len(result_per)):
    #     print(res[idx])

    count = {}
    for idx in result:
        try:
            count[idx] += 1
        except:
            count[idx] = 1
    print(count)
    res = sorted(count.items(), key=(lambda x: x[1]), reverse=True)

    for idx in range(len(res)):
        if idx < 5:
            name = res[idx][0]
            count = res[idx][1]
            print(name, count)
        idx += 1

    name = res[0][0].split('_')
    name = name[0]

    text = ax.text(rects[i][0][0], rects[i][0][1], name, color='b', fontsize=15, fontweight='bold')
    text.set_path_effects([path_effects.Stroke(linewidth=10, foreground='white'), path_effects.Normal()])
    rect = patches.Rectangle(rects[i][0],
                             rects[i][1][1] - rects[i][0][1],
                             rects[i][1][0] - rects[i][0][0],
                             linewidth=2, edgecolor='w', facecolor='none')

    ax.add_patch(rect)

plt.axis('off')
# plt.savefig('result/output.png')
plt.show()