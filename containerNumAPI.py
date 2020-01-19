from keras import backend as K
from keras.models import load_model
from models.keras_ssd7 import build_model
from keras.preprocessing import image

from matplotlib import pyplot as plt
from imageio import imread
import numpy as np
import cv2

import urllib.request
import urllib.parse
import json
import time
import base64

import ssl
from flask import Flask, request
from flask_restful import reqparse, Api, Resource

ssl._create_default_https_context = ssl._create_unverified_context


class containerNumPredict():

    def __init__(self, image_dir, appcode, url_request):

        self.img_height = 300  # Height of the input images
        self.img_width = 480  # Width of the input images
        self.img_channels = 3  # Image channels
        self.n_classes = 1  # number of class user need to detect.
        self.confi_threshold = 0.2  # parameter for confidence threshold to yeild potential object.
        self.appcode = 'APPCODE ' + appcode  # Appcode for OCR.
        self.image_dir = image_dir  # directory of image file.
        self.url_request = url_request  # url_request for OCR.
        self.headers = {
            'Authorization': self.appcode,
            'Content-Type': 'application/json; charset=UTF-8'
        }  # header param for OCR
        self.xmin = 0
        self.ymin = 0
        self.xmax = 0
        self.ymax = 0

    def predictObject(self):

        K.clear_session()  # Clear previous models from memory.

        # build object detection model
        model = build_model(image_size=(self.img_height, self.img_width, self.img_channels),
                            n_classes=self.n_classes,
                            mode='inference',
                            l2_regularization=0.0005,
                            scales=[0.08, 0.16, 0.32, 0.64, 0.96],
                            aspect_ratios_global=[0.5, 1.0, 2.0],
                            aspect_ratios_per_layer=None,
                            two_boxes_for_ar1=True,
                            steps=None,
                            offsets=None,
                            clip_boxes=False,
                            variances=[1.0, 1.0, 1.0, 1.0],
                            normalize_coords=True,
                            subtract_mean=127.5,
                            divide_by_stddev=127.5)

        # 2: Optional: Load some weights
        model.load_weights('./trained_weights_ssd7/ssd7_epoch-65_loss-2.1049_val_loss-1.9071.h5', by_name=True)

        # Two list for storing the original image data and resized image data
        orig_images = []  # Store the original images here.
        input_images = []  # Store the resized images here.

        # We'll only load one image in this example.
        print(self.image_dir);
        orig_images.append(imread(self.image_dir))
        img = image.load_img(self.image_dir, target_size=(self.img_height, self.img_width))
        img = image.img_to_array(img)
        input_images.append(img)
        input_images = np.array(input_images)

        # predicting image
        y_pred = model.predict(input_images)

        # using confidence rate to filter the prediction output
        y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > self.confi_threshold] for k in range(y_pred.shape[0])]

        if len(y_pred_thresh) == 1 and len(y_pred_thresh[0]) == 0:
            return 600  # code 600 means a new image is needed
        #            raise ValueError("Object detection failed. Please reupload a new image!")

        max_prob = 0

        for box in y_pred_thresh[0]:
            if box[1] > max_prob:
                max_prob = box[1]
        print(max_prob)

        for box in y_pred_thresh[0]:
            if box[1] == max_prob:
                # Transform the predicted bounding boxes for the 300x480 image to the original image dimensions.
                self.xmin = int(box[2] * orig_images[0].shape[1] / self.img_width)
                self.ymin = int(box[3] * orig_images[0].shape[0] / self.img_height)
                self.xmax = int(box[4] * orig_images[0].shape[1] / self.img_width) + 20
                self.ymax = int(box[5] * orig_images[0].shape[0] / self.img_height) + 20

        print(self.xmin, self.ymin, self.xmax, self.ymax)

        # The part of image cropped for containerNum.
        cropImg = orig_images[0][self.ymin:self.ymax, self.xmin:self.xmax]

        if cropImg is not None:
            print("Object detection successfully!")
        else:
            return 600  # code 600 means a new image is needed
        #            raise ValueError("Object detection failed. Please reupload a new image!")
        #       cv2.imwrite("./test_data/检测3117"+".jpg", cropImg)

        # with open('/Users/jrr/Downloads/object_detection/test_data/target19.jpg', 'rb') as f:  # 以二进制读取本地图片
        #     data = f.read()
        #     encodestr = str(base64.b64encode(data), 'utf-8')
        # self.diction = {'img': encodestr}

        img_encode = cv2.imencode('.jpg', cropImg)[1]
        encodestr = str(base64.b64encode(img_encode), 'utf-8')
        diction = {'img': encodestr}

        return self.posturl(self.url_request, diction)

    #     def openObjectimage(self):
    #         with open('./test_data/检测3117.jpg', 'rb') as f:  # 以二进制读取本地图片
    #             data = f.read()

    #             encodestr = str(base64.b64encode(data),'utf-8')

    #         return encodestr

    def posturl(self, url, data={}):
        try:
            params = json.dumps(data).encode(encoding='UTF8')
            req = urllib.request.Request(url, params, self.headers)
            r = urllib.request.urlopen(req)
            html = r.read()
            r.close();
            return html.decode("utf8")
        except urllib.error.HTTPError as e:
            return (e.code)
            print(e.code)
            print(e.read().decode("utf8"))
        time.sleep(1)

    def rectifyResult(self, ocr_str):
        res = ''
        for i in range(len(ocr_str) - 7):
            if ocr_str[i:i + 7] == '"word":':
                for j in range(i + 8, len(ocr_str) - 7):
                    if ocr_str[j] == '"':
                        res += ocr_str[i + 8:j]
                        break

        res = res.replace(' ', '')
        res = res[:11]

        if len(res) < 10:
            return 600
        # to list, so that easy to revise
        res_list = list(res)

        for i in range(4):
            if res_list[i] == '0':
                res_list[i] = 'O'
        for i in range(4, len(res_list)):
            if res_list[i] == 'o' or res_list[i] == 'O' or res_list[i] == 'Q':
                res_list[i] = '0'
            if res_list[i] == 'g':
                res_list[i] = '9'

        # check digit
        # create dict for character
        dic = {str(i): i for i in range(10)}
        num = 10
        for i in range(26):
            dic[chr(65 + i)] = num
            num += 1
            if not num % 11:
                num += 1
        # calculate check digit
        tmp_sum = 0
        for i in range(10):
            tmp_sum += dic[res_list[i]] * (2 ** i)
        check_digit = tmp_sum % 11
        if check_digit == 10:
            check_digit = 0

        if len(res_list) < 11:
            res_list.append(str(check_digit))
        if res_list[-1] != str(check_digit) or res_list[-1].isalpha():
            print('check digit incorrect!')
            res_list[-1] = str(check_digit)

        res = ''.join(res_list)
        return res


parser_put = reqparse.RequestParser()
parser_put.add_argument("path", type=str, required=True, help="need image path")
app = Flask(__name__)

api = Api(app)


@app.route('/recognition', methods=['get', 'post'])
def recognition():
    if 'path' in request.args:
        infos = request.args['path']
    elif 'path' in request.json:
        infos = request.json['path']
    else:
        tmp = parser_put.parse_args()
        infos = tmp['path']
    # infos = request.args['path']

    # tmp = parser_put.parse_args()
    # print("得到的request:" + str(tmp))
    # infos = tmp['path']

    # image_dir = "./test/image011.JPG"
    url_request = "https://ocrapi-advanced.taobao.com/ocrservice/advanced"
    appcode = '4335919ca03a4714832942f64c6cef2b'

    # infos = "/Users/jrr/Downloads/object_detection/test/image606.jpg"
    containerNum = containerNumPredict(image_dir=infos, appcode=appcode, url_request=url_request)

    ocr_str = containerNum.predictObject()
    # print(ocr_str)
    if ocr_str == 400:
        res = "阿里OCR参数错误"
    elif ocr_str == 401:
        res = "您无阿里OCR的权限，请开通后使用"
    elif ocr_str == 403:
        res = "阿里OCR购买的容量已用完或者签名错误"
    elif ocr_str == 500:
        res = "阿里OCR服务器错误，请稍后重试"
    elif ocr_str == 600:
        res = "无法识别该图片，请重新上传图片"
    else:
        res = containerNum.rectifyResult(ocr_str)
    return res


if __name__ == "__main__":
    # image_dir = "./test/image011.JPG"
    # url_request = "https://ocrapi-advanced.taobao.com/ocrservice/advanced"
    # appcode = '4335919ca03a4714832942f64c6cef2b'
    #
    # containerNum = containerNumPredict(image_dir=image_dir, appcode=appcode, url_request = url_request)
    # ocr_str = containerNum.predictObject()
    # res = containerNum.rectifyResult(ocr_str)
    #
    app.debug = True
    app.run(host='0.0.0.0', port=8222)
    # print(res)
