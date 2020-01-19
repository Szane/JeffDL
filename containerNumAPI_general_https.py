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
from flask_cors import CORS

import re

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

        self.firstTry = True  # 给两次搜索的机会

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
        self.orig_images = []  # Store the original images here.
        input_images = []  # Store the resized images here.

        # We'll only load one image in this example.
        resp = urllib.request.urlopen(self.image_dir)
        img_orig = np.asarray(bytearray(resp.read()), dtype="uint8")
        img_orig = cv2.imdecode(img_orig, cv2.IMREAD_COLOR)

        self.orig_images.append(img_orig)
        #        img = image.load_img(img_orig, target_size=(self.img_height, self.img_width))
        #        img = image.img_to_array(img)
        #        input_images.append(img)
        #        input_images = np.array(input_images)
        img_new = cv2.resize(img_orig, (self.img_width, self.img_height), interpolation=cv2.INTER_CUBIC)
        input_images.append(img_new)
        input_images = np.array(input_images)
        # predicting image
        y_pred = model.predict(input_images)

        # using confidence rate to filter the prediction output
        y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > self.confi_threshold] for k in range(y_pred.shape[0])]

        if len(y_pred_thresh) == 1 and len(y_pred_thresh[0]) == 0:
            self.firstTry = False
            img_encode = cv2.imencode('.jpg', self.orig_images[0])[1]
            encodestr = str(base64.b64encode(img_encode), 'utf-8')
            diction = {'image': encodestr}
            return self.posturl(self.url_request, diction)

            # return 600 # code 600 means a new image is needed

        #            raise ValueError("Object detection failed. Please reupload a new image!")

        max_prob = 0

        for box in y_pred_thresh[0]:
            if box[1] > max_prob:
                max_prob = box[1]
        #        print(max_prob)

        for box in y_pred_thresh[0]:
            if box[1] == max_prob:
                # Transform the predicted bounding boxes for the 300x480 image to the original image dimensions.
                self.xmin = max(0, int(box[2] * self.orig_images[0].shape[1] / self.img_width) - 100)
                self.ymin = max(0, int(box[3] * self.orig_images[0].shape[0] / self.img_height) - 20)
                self.xmax = min(int(box[4] * self.orig_images[0].shape[1] / self.img_width) + 90,
                                self.orig_images[0].shape[1])
                self.ymax = min(int(box[5] * self.orig_images[0].shape[0] / self.img_height) + 20,
                                self.orig_images[0].shape[0])

        #        print(self.xmin, self.ymin, self.xmax, self.ymax)

        # The part of image cropped for containerNum.
        cropImg = self.orig_images[0][self.ymin:self.ymax, self.xmin:self.xmax]

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
        diction = {'image': encodestr}

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

    def extractWordsFromOCRResult(self, ocr_str):
        data_dic = json.loads(ocr_str)
        #        print(data_dic)
        candidates = []
        if 'ret' not in data_dic:
            return 700
        for pair in data_dic['ret']:
            candidates.append([pair['rect']['top'], pair['rect']['left'], \
                               pair['word'], pair['rect']['height'], pair['rect']['width']])

        if not candidates:
            return 700

        candidates.sort(key=lambda x: x[0] + x[1])

        if len(candidates) == 1:
            return candidates[0][2]

        if len(candidates) < 8:  # 没有全图搜的情况，特征数没有那么多
            for c in candidates:
                if len(c[2]) >= 10:
                    if c[2].find('<unk>') >= 0:
                        c[2] = c[2].replace('<unk>', '')
                    elif len(c[2].replace(' ', '')) >= 10:
                        return c[2]

                if len(c[2]) >= 6:
                    # if len(c[2]) >= 8 and len(c[2].replace(" ", "")) >= 8 and c[2][0] == '1':
                    #     return candidates[0][2] + c[2].replace(" ", "")[1:]
                    #                    print("这儿")
                    #                    print(candidates[0][2])
                    #                    print(c[2])
                    return candidates[0][2] + c[2]

        i = 1
        while i < len(candidates):
            if ' ' in candidates[i][2]:
                strs = candidates[i][2].split(' ')
                if candidates[i - 1][2].find(strs[0]) >= 0:
                    candidates[i][2] = ''.join(strs[1:])
            # 如果该结果为之前结果的子串，删除掉。例如之前识别了ABCD, 此时出现了BCD，显然OCR识别重复了。
            if len(candidates[i][2]) > 1 and candidates[i - 1][2].find(candidates[i][2]) >= 0:
                candidates.pop(i)
            else:
                i += 1

        res = ''
        for c in candidates:
            res += c[2]
            if len(res) >= 10:
                break
        #        print(res)
        return res

    def checkIfTryAgain(self):
        if not self.firstTry:
            return 700
        self.firstTry = False
        img_encode = cv2.imencode('.jpg', self.orig_images[0])[1]
        encodestr = str(base64.b64encode(img_encode), 'utf-8')
        diction = {'image': encodestr}
        ocr_str = self.posturl(self.url_request, diction)
        res = self.extractWordsFromOCRResult(ocr_str)
        return self.rectifyResult(res)

    def rectifyResult(self, res):

        # res = res.replace(' ', '')
        # res = res.replace('.', '')
        res = re.sub("[^0-9a-zA-Z]", "", res)
        res = res[:11]

        if len(res) < 10:
            return self.checkIfTryAgain()

        # to list, so that easy to revise
        res_list = list(res)

        for i in range(4):
            if res_list[i] == '0':
                res_list[i] = 'O'
            if res_list[i] == 'u':
                res_list[i] = 'U'

        for i in range(4, len(res_list)):
            if res_list[i] == 'o' or res_list[i] == 'O' or res_list[i] == 'Q':
                res_list[i] = '0'
            elif res_list[i] == 'g':
                res_list[i] = '9'
            elif res_list[i] == 'B':
                res_list[i] = '8'
            elif res_list[i] == 'I':
                res_list[i] = '1'
            elif res_list[i] == 'A':
                res_list[i] = '4'

        if not res_list[3].isalpha():  # 只读取了前三个字母，框的区域短了一点
            return self.checkIfTryAgain()

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
            if res_list[i] not in dic:
                return self.checkIfTryAgain()
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
        #        print(res)
        return res


parser_put = reqparse.RequestParser()
parser_put.add_argument("path", type=str, required=True, help="need image path")
app = Flask(__name__)
CORS(app, supports_credentials=True)
api = Api(app)


@app.route('/recognition', methods=['get', 'post'])
def recognition():
    if 'path' in request.args:
        infos = request.args['path']
    else:
        tmp = parser_put.parse_args()
        infos = tmp['path']
    # infos = request.args['path']

    # tmp = parser_put.parse_args()
    # print("得到的request:" + str(tmp))
    # infos = tmp['path']

    # image_dir = "./test/image011.JPG"
    url_request = "https://tysbgpu.market.alicloudapi.com/api/predict/ocr_general"
    appcode = '4335919ca03a4714832942f64c6cef2b'

    # infos = "/Users/jrr/Downloads/object_detection/test/image606.jpg"
    containerNum = containerNumPredict(image_dir=infos, appcode=appcode, url_request=url_request)

    ocr_str = containerNum.predictObject()
    #    print(ocr_str)
    success = 0
    msg = ''
    if ocr_str == 400:
        msg = "阿里OCR参数错误"
    elif ocr_str == 401:
        msg = "您无阿里OCR的权限，请开通后使用"
    elif ocr_str == 403:
        msg = "阿里OCR购买的容量已用完或者签名错误"
    elif ocr_str == 500:
        msg = "阿里OCR服务器错误，请稍后重试"
    elif ocr_str == 600:
        msg = "无法识别该图片，请重新上传图片"
    elif ocr_str == 700:
        msg = "解析出错，请重新上传图片"
    else:
        res = containerNum.extractWordsFromOCRResult(ocr_str)
        if res == 700:
            msg = "解析出错，请重新上传图片"
            return json.dumps({"res": res, "msg": msg, "success": success})
        else:
            res = containerNum.rectifyResult(res)
            if res == 700:
                msg = "解析出错，请重新上传图片"
                return json.dumps({"res": res, "msg": msg, "success": success})
            else:
                success = 1
    return json.dumps({"res": res, "msg": msg, "success": success})


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
