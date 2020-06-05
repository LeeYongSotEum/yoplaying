import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

img_paths = {}
descs = {}
# 강동원
for i in range(132):
    name = 'KangDongWon_'+ str(i+1)
    path = 'image/KangDongWon/KangDongWon ('+ str(i+1) +').jpg'
    img_paths.update({name:path})
    descs.update({name:None})

# 차은우
for i in range(116):
    name = 'ChaEunWoo_'+ str(i+1)
    path = 'image/ChaEunWoo/ChaEunWoo ('+ str(i+1) +').jpg'
    img_paths.update({name:path})
    descs.update({name:None})

# 차승원
for i in range(119):
    name = 'ChaSeungWon_'+ str(i+1)
    path = 'image/ChaSeungWon/ChaSeungWon ('+ str(i+1) +').jpg'
    img_paths.update({name:path})
    descs.update({name:None})

# 공유
for i in range(116):
    name = 'GongYoo_'+ str(i+1)
    path = 'image/GongYoo/GongYoo ('+ str(i+1) +').jpg'
    img_paths.update({name:path})
    descs.update({name:None})

# 고수
for i in range(116):
    name = 'GoSu_'+ str(i+1)
    path = 'image/GoSu/GoSu ('+ str(i+1) +').jpg'
    img_paths.update({name:path})
    descs.update({name:None})

# 현빈
for i in range(115):
    name = 'HyunBin_'+ str(i+1)
    path = 'image/HyunBin/HyunBin ('+ str(i+1) +').jpg'
    img_paths.update({name:path})
    descs.update({name:None})

# 장동건
for i in range(119):
    name = 'JangDongGun_'+ str(i+1)
    path = 'image/JangDongGun/JangDongGun ('+ str(i+1) +').jpg'
    img_paths.update({name:path})
    descs.update({name:None})

# 장동민
for i in range(119):
    name = 'JangDongMin_'+ str(i+1)
    path = 'image/JangDongMin/JangDongMin ('+ str(i+1) +').jpg'
    img_paths.update({name:path})
    descs.update({name:None})

# 정준하
for i in range(94):
    name = 'JungJoonHa_'+ str(i+1)
    path = 'image/JungJoonHa/JungJoonHa ('+ str(i+1) +').jpg'
    img_paths.update({name:path})
    descs.update({name:None})

# 정우성
for i in range(114):
    name = 'JungWooSung_'+ str(i+1)
    path = 'image/JungWooSung/JungWooSung ('+ str(i+1) +').jpg'
    img_paths.update({name:path})
    descs.update({name:None})

# 강호동
for i in range(109):
    name = 'KangHoDong_'+ str(i+1)
    path = 'image/KangHoDong/KangHoDong ('+ str(i+1) +').jpg'
    img_paths.update({name:path})
    descs.update({name:None})

# 김종국
for i in range(113):
    name = 'KimJongKook_'+ str(i+1)
    path = 'image/KimJongKook/KimJongKook ('+ str(i+1) +').jpg'
    img_paths.update({name:path})
    descs.update({name:None})

# 김요한
for i in range(135):
    name = 'KimYoHan_'+ str(i+1)
    path = 'image/KimYoHan/KimYoHan ('+ str(i+1) +').jpg'
    img_paths.update({name:path})
    descs.update({name:None})

# 이병헌
for i in range(124):
    name = 'LeeByungHun_'+ str(i+1)
    path = 'image/LeeByungHun/LeeByungHun ('+ str(i+1) +').jpg'
    img_paths.update({name:path})
    descs.update({name:None})

# 이진호
for i in range(110):
    name = 'LeeJinHo_'+ str(i+1)
    path = 'image/LeeJinHo/LeeJinHo ('+ str(i+1) +').jpg'
    img_paths.update({name:path})
    descs.update({name:None})

# 이광수
for i in range(118):
    name = 'LeeKwangSoo_'+ str(i+1)
    path = 'image/LeeKwangSoo/LeeKwangSoo ('+ str(i+1) +').jpg'
    img_paths.update({name:path})
    descs.update({name:None})

# 남궁민
for i in range(104):
    name = 'NamGungMin_'+ str(i+1)
    path = 'image/NamGungMin/NamGungMin ('+ str(i+1) +').jpg'
    img_paths.update({name:path})
    descs.update({name:None})

# 박명수
for i in range(118):
    name = 'ParkMyeongSoo_'+ str(i+1)
    path = 'image/ParkMyeongSoo/ParkMyeongSoo ('+ str(i+1) +').jpg'
    img_paths.update({name:path})
    descs.update({name:None})

# 박서준
for i in range(112):
    name = 'ParkSeoJoon_'+ str(i+1)
    path = 'image/ParkSeoJoon/ParkSeoJoon ('+ str(i+1) +').jpg'
    img_paths.update({name:path})
    descs.update({name:None})

# 서강준
for i in range(110):
    name = 'SeoKangJoon_'+ str(i+1)
    path = 'image/SeoKangJoon/SeoKangJoon ('+ str(i+1) +').jpg'
    img_paths.update({name:path})
    descs.update({name:None})

# 신동엽
for i in range(118):
    name = 'SinDongYeop_'+ str(i+1)
    path = 'image/SinDongYeop/SinDongYeop ('+ str(i+1) +').jpg'
    img_paths.update({name:path})
    descs.update({name:None})

# 성시경
for i in range(104):
    name = 'SungSiKyung_'+ str(i+1)
    path = 'image/SungSiKyung/SungSiKyung ('+ str(i+1) +').jpg'
    img_paths.update({name:path})
    descs.update({name:None})

# 뷔
for i in range(117):
    name = 'V_'+ str(i+1)
    path = 'image/V/V ('+ str(i+1) +').jpg'
    img_paths.update({name:path})
    descs.update({name:None})

# 원빈
for i in range(99):
    name = 'WonBin_'+ str(i+1)
    path = 'image/WonBin/WonBin ('+ str(i+1) +').jpg'
    img_paths.update({name:path})
    descs.update({name:None})

# 양세찬
for i in range(108):
    name = 'YangSeChan_'+ str(i+1)
    path = 'image/YangSeChan/YangSeChan ('+ str(i+1) +').jpg'
    img_paths.update({name:path})
    descs.update({name:None})

# 양세형
for i in range(114):
    name = 'YangSeHyung_'+ str(i+1)
    path = 'image/YangSeHyung/YangSeHyung ('+ str(i+1) +').jpg'
    img_paths.update({name:path})
    descs.update({name:None})

# 유재석
for i in range(90):
    name = 'YooJaeSuk_'+ str(i+1)
    path = 'image/YooJaeSuk/YooJaeSuk ('+ str(i+1) +').jpg'
    img_paths.update({name:path})
    descs.update({name:None})

# 유세윤
for i in range(138):
    name = 'YooSeYoon_'+ str(i+1)
    path = 'image/YooSeYoon/YooSeYoon ('+ str(i+1) +').jpg'
    img_paths.update({name:path})
    descs.update({name:None})
#
# #
# for i in range():
#     name = '_'+ str(i+1)
#     path = 'image// ('+ str(i+1) +').jpg'
#     img_paths.update({name:path})
#     descs.update({name:None})





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
    print(name)
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    _, img_shapes, _ = find_faces(img_rgb)
    if img_shapes == 0:
        print("못찾은 이미지 :", name)
    if img_shapes == 2:
        print('얼굴이 2개 이상 :', name)
    else:
        descs[name] = encode_faces(img_rgb, img_shapes)[0]
        # pass
    # print(descs[name])

np.save('image/descs.npy', descs)
# print(descs)
print('Done')