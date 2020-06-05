from flask import Flask, render_template, jsonify, request, Response, send_file
from datetime import datetime
import dlib
import io
import cv2
import pandas as pd
import numpy as np
from flask_frozen import Freezer

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')
descs = np.load('image/descs.npy', allow_pickle=True).item()


app = Flask(__name__)
# freezer = Freezer(app)

posts = [
    {
        'author': {
            'username': 'test-user'
        },
        'title': '첫 번째 포스트',
        'content': '첫 번째 포스트 내용입니다.',
        'date_posted': datetime.strptime('2020-03-01', '%Y-%m-%d')
    },
    {
        'author': {
            'username': 'test-user'
        },
        'title': '두 번째 포스트',
        'content': '두 번째 포스트 내용입니다.',
        'date_posted': datetime.strptime('2020-04-03', '%Y-%m-%d')
    },
]













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

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', posts=posts)

@app.route('/about')
def about():
    return render_template('about.html', title='About')

@app.route('/lotto_number')
def lotto_number():
    return render_template('lotto_number.html', title='Lotto')

@app.route('/get_lotto_number')
def get_lotto_number():
    df_train = pd.read_csv('templates/input/lotto_data_train.csv', encoding='euc-kr')
    # df_train = pd.read_csv('C:/Users/LYSE/Desktop/Study/Human_Face_Detector/templates/input/lotto_data_train.csv', encoding='euc-kr')
    df_train = df_train.drop(['Round'], axis=1)
    df_num1 = df_train['Num1']
    df_num2 = df_train['Num2']
    df_num3 = df_train['Num3']
    df_num4 = df_train['Num4']
    df_num5 = df_train['Num5']
    df_num6 = df_train['Num6']

    df_train_list = [df_num1, df_num2, df_num3, df_num4, df_num5, df_num6]
    choice_number = []

    for train in df_train_list:
        values = train.value_counts().sort_index().keys().tolist()
        counts = train.value_counts([0]).sort_index().tolist()
        idx = len(choice_number)
        if idx == 0:
            number = np.random.choice(values, p=counts)
            choice_number.append(int(number))
        else:
            while (1):
                if choice_number[-1] >= number:
                    number = np.random.choice(values, p=counts)
                    continue
                elif choice_number[-1] < number:
                    choice_number.append(int(number))
                    break

    print(choice_number)
    return jsonify(choice_number=choice_number)

@app.route('/face_detector')
def face_detector():
    return render_template('face_detector.html', title='Face')

@app.route('/404.html')
def error404():
    return render_template('404.html')

@app.route('/get_data', methods=['POST'])
def get_data():

    if request.method == 'POST':
        print('POST')
        _file = request.files.get('file')
        in_memory_file = io.BytesIO()
        _file.save(in_memory_file)
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        color_image_flag = 1
        img_bgr = cv2.imdecode(data, color_image_flag)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rects, shapes, _ = find_faces(img_rgb)

        if len(shapes) == 0 :
            pass
        elif len(shapes) >= 2:
            pass
        else:
            descriptors = encode_faces(img_rgb, shapes)
            #
            # fig, ax = plt.subplots(1, figsize=(10, 10))
            # ax.imshow(img_rgb)

            for i, desc in enumerate(descriptors):
                dist_dic = {}
                for name, saved_desc in descs.items():
                    dist = np.linalg.norm([desc] - saved_desc, axis=1)
                    dist = 1 - dist
                    dist_dic.update({name: dist})

                res = sorted(dist_dic.items(), key=(lambda x: x[1]), reverse=True)
                print(res)

                result = []
                for idx in range(100):
                    temp = res[idx][0].split('_')[0]
                    result.append(temp)

                count = {}
                for idx in result:
                    try:
                        count[idx] += 1
                    except:
                        count[idx] = 1
                # print(count)
                res = sorted(count.items(), key=(lambda x: x[1]), reverse=True)

                name_list = []
                count_list = []
                for idx in range(len(res)):
                    if idx < 5:
                        name = res[idx][0]
                        count = res[idx][1]
                        name_list.append(name)
                        count_list.append(count)
                        print(name, count)
                    idx += 1

                print(name_list)
                print(count_list)

                # name = res[0][0].split('_')
                # name = name[0]

                # text = ax.text(rects[i][0][0], rects[i][0][1], name, color='b', fontsize=15, fontweight='bold')
                # text.set_path_effects([path_effects.Stroke(linewidth=10, foreground='white'), path_effects.Normal()])
                # rect = patches.Rectangle(rects[i][0],
                #                          rects[i][1][1] - rects[i][0][1],
                #                          rects[i][1][0] - rects[i][0][0],
                #                          linewidth=2, edgecolor='w', facecolor='none')

                # ax.add_patch(rect)


            # plt.axis('off')
            # plt.show()

    return jsonify(result_name=name_list, result_count=count_list)

        # in_memory_file = io.BytesIO()
        # FigureCanvasAgg(fig).print_png(in_memory_file)

        # plt.savefig(in_memory_file, format='png')
        # in_memory_file.seek(0)
        # plot_url = base64.b64decode(in_memory_file.getvalue())
        # print(in_memory_file)
        #
        # mystr = '<img src="data:image/png;base64,{}">'.format(plot_url)

    # return jsonify({'url':mystr})
    # return render_template('face_detector.html')
    # return Response(in_memory_file.getvalue(), mimetype='image/png')
    # return send_file(in_memory_file.getvalue(), mimetype='image/png')
    # return 'OK'

if __name__ == '__main__':
    # if len(sys.argv) > 1 and sys.argv[1] == 'build':
    #     print("Building Website...")
    #     freezer.freeze()
    # else:
    app.run(debug=True)
