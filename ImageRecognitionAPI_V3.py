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

import requests as req
from PIL import Image
from io import BytesIO

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
        try:
            resp = urllib.request.urlopen(self.image_dir)
        except urllib.error.HTTPError as e:
            print("识别图片URL路径错误！")
            return [e.code, e.code]

        img_orig = np.asarray(bytearray(resp.read()), dtype="uint8")
        img_orig = cv2.imdecode(img_orig, cv2.IMREAD_COLOR)

        # self.orig_images.append(img_orig)
        self.orig_images.append(img_orig)

        # img_new = cv2.resize(img_orig, (self.img_width, self.img_height), interpolation=cv2.INTER_CUBIC)
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
            return [self.posturl(self.url_request, diction), self.posturl(self.url_request, diction)]

        max_prob = 0

        for box in y_pred_thresh[0]:
            if box[1] > max_prob:
                max_prob = box[1]

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
        cropImg_conNum = self.orig_images[0][self.ymin:self.ymax, self.xmin:self.xmax]
        # cropImg_weight = self.orig_images[0][min(int(box[5] * self.orig_images[0].shape[0] / self.img_height) + 80,
        #                         self.orig_images[0].shape[0]):self.orig_images[0].shape[0], self.xmin:self.orig_images[0].shape[1]]
        cropImg_weight = self.orig_images[0][min(self.ymax + 60, self.orig_images[0].shape[0]):self.orig_images[0].shape[0], self.xmin:self.orig_images[0].shape[1]]

        if cropImg_conNum is not None:
            print("Object detection successfully!")
        else:
            return [600,600]  # code 600 means a new image is needed


        img_encode_conNum = cv2.imencode('.jpg', cropImg_conNum)[1]
        encodestr_conNum = str(base64.b64encode(img_encode_conNum), 'utf-8')
        
        diction_conNum = {'image': encodestr_conNum}

        img_encode_weight = cv2.imencode('.jpg', cropImg_weight)[1]
        encodestr_weight = str(base64.b64encode(img_encode_weight), 'utf-8')
        
        diction_weight = {'image': encodestr_weight}

        result_conNum = self.posturl(self.url_request, diction_conNum)
        result_weight = self.posturl(self.url_request, diction_weight)

        # return [self.posturl(self.url_request, diction_conNum), self.posturl(self.url_request, diction_weight)]
        return [result_weight, result_conNum]


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

    def extractWeightOCR(self, weightOCR):
        words = []
        numbers = []
    # 从箱号下方区域中提取文字信息
        candidates = []

 #       print(weightOCR)
 #       print(type(weightOCR))

        data_dic = json.loads(weightOCR)
        if 'ret' not in data_dic:
            return 700

        for pair in data_dic['ret']:
            candidates.append([pair['rect']['top'], pair['word']])

        if not candidates:
            return 700

        candidates.sort(key = lambda c: c[0])

        for i in candidates:
            words.append(i[1])

        pattern = re.compile(r'\d+')

        print(words)

        for word in words:
            if len(word) > 12:
                if 'KG' in word or 'K.G' in word:
                    numbers += re.split(r'[a-zA-Z]+', word)
            else:
                res = re.findall(pattern, word)
                num = ''
                for i in res:
                    num += i
                if num:
                    numbers.append(num)

        if len(numbers) != 0:
            for i in range(len(numbers)):
                numbers[i] = ''.join(list(filter(str.isdigit, numbers[i])))

            while "" in numbers:
                numbers.remove("")
            
            while " " in numbers:
                numbers.remove(" ")

        print(numbers)
        
        # if len(numbers) > 6:
        #     numbers.sort(key = lambda c: int(c))
        for i in range(len(numbers)):
            if len(numbers[i]) == 4:
                for val in range(-20, 21):
                    if str(round(int(numbers[i]) * 2.2046) + val) in numbers:
                        return [numbers[i]+'KG', str(round(int(numbers[i]) * 2.2046) + val) + 'LB']
            
        numbers.sort(key = lambda c: int(c))
        
        if len(numbers) > 6:
            for c in numbers[::-1]:
                if len(c) == 4:
                    if int(c) < 5500:
                        return [str(c) + 'KG', str(round(int(c) * 2.2046))+'LB']
                    # return [c+'LB', numbers]
                    else:
                        return [str(round(int(c) / 2.2046)) + 'KG', c+'LB']

        if len(numbers) <= 6:
            for c in numbers:
                if len(c) == 4:
                    return [c+'KG', str(round(int(c) * 2.2046))+'LB']

        print('numbers: ' + str(numbers))
            
            # numbers_combine.sort(key = lambda c: int(c))

        # if len(numbers) :
        #     return [numbers[2], numbers]
        if len(numbers) == 0:
            return 700

        return ["", ""]

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

        print(candidates)
        
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
        return res

class API_mashiji():

    def __init__(self, image_dir, appcode, url_request):
        self.appcode = 'APPCODE ' + appcode  # Appcode for OCR.
        self.image_dir = image_dir  # directory of image file.
        self.url_request = url_request  # url_request for OCR.
        self.headers = {
            'Authorization': self.appcode,
            'Content-Type': 'application/json; charset=UTF-8'
        }

    def imageToOCR(self):
        with open(self.image_dir, 'rb') as f:  # 以二进制读取本地图片
            data = f.read()
            encodestr = str(base64.b64encode(data), 'utf-8')
        return encodestr

    def urlImage(self):
        resp = urllib.request.urlopen(self.image_dir)
        encodestr = np.asarray(bytearray(resp.read()), dtype="uint8")
        #encodestr = cv2.imdecode(encodestr, cv2.IMREAD_COLOR)
        encodestr = str(base64.b64encode(encodestr), 'utf-8')
        return encodestr

    def posturl(self, data={}):
        try:
            params = json.dumps(data).encode(encoding='UTF8')
            req = urllib.request.Request(self.url_request, params, self.headers)
            r = urllib.request.urlopen(req)
            html = r.read()
            r.close()
            return html.decode("utf8")
        except urllib.error.HTTPError as e:
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
            output = candidates[0][2].replace(' ','')

            return output

#        if len(candidates) < 8:  # 没有全图搜的情况，特征数没有那么多
        for c in candidates:
            if 13 >= len(c[2]) >= 7:
                if c[2].find('<unk>') >= 0:
                    c[2] = c[2].replace('<unk>', '')
                output = c[2].replace(' ', '')
                if len(output) >= 7 and len(''.join(re.findall(r'\d*', output))) >= 7:
#                    print('diyi')
                    return output

                if len(output) >= 9 and len(''.join(re.findall(r'\d*', output))) >= 6:
                    if output[0].isalpha():
#                        print("dier")
                        return output

                if len(output) >= 8 and len(''.join(re.findall(r'\d*', output))) >= 4:
                    if output[0].isalpha():
#                        print("disan")
                        return output

                if len(output) >= 9 and len(''.join(re.findall(r'\d*', output))) >= 3:
                    if output[0].isalpha():
#                        print("disi")
                        return output

        # i = 1
        # while i < len(candidates):
        #     if ' ' in candidates[i][2]:
        #         strs = candidates[i][2].split(' ')
        #         if candidates[i - 1][2].find(strs[0]) >= 0:
        #             candidates[i][2] = ''.join(strs[1:])
        #     # 如果该结果为之前结果的子串，删除掉。例如之前识别了ABCD, 此时出现了BCD，显然OCR识别重复了。
        #     if len(candidates[i][2]) > 1 and candidates[i - 1][2].find(candidates[i][2]) >= 0:
        #         candidates.pop(i)
        #     else:
        #         i += 1
        #
        for c in candidates:
            if 13 >= len(c[2]) >= 7:
                if c[2].find('<unk>') >= 0:
                    c[2] = c[2].replace('<unk>', '')
                output = c[2].replace(' ', '')
                if len(output) >= 5 and len(''.join(re.findall(r'\d*', output))) >= 5:
                    return output

        # if re.match(r'^[A-Za-z0-9]+', res):
        #     return res
        # else:
        #     return 700
        return 700


parser_put = reqparse.RequestParser()
parser_put.add_argument("path", type=str, required=True, help="need image path")
app = Flask(__name__)
CORS(app, supports_credentials=True)
api = Api(app)

#提取集装箱号的app route.
@app.route('/container', methods=['get', 'post'])
def container():
    if 'path' in request.args:
        infos = request.args['path']
    else:
        tmp = parser_put.parse_args()
        infos = tmp['path']

    url_request = "https://tysbgpu.market.alicloudapi.com/api/predict/ocr_general"
    appcode = '4335919ca03a4714832942f64c6cef2b'

    containerNum = containerNumPredict(image_dir=infos, appcode=appcode, url_request=url_request)

    ocr_str = containerNum.predictObject()
    ocr_str_conNum = ocr_str[1]
    ocr_str_weight = ocr_str[0]

    success_conNum = 0
    success_weight = 0
    msg_conNum = ""
    msg_weight = ""
    res_conNum = ""
    res_weight_kg = ""
    res_weight_lb = ""

    if ocr_str_conNum == 400:
        msg_conNum = "阿里OCR参数错误"
        print("箱号识别：阿里OCR参数错误")
    elif ocr_str_conNum == 401:
        msg_conNum = "您无阿里OCR的权限，请开通后使用"
        print("箱号识别：您无阿里OCR的权限，请开通后使用")
    elif ocr_str_conNum == 403:
        msg_conNum = "阿里OCR购买的容量已用完或者签名错误"
        print("箱号识别:阿里OCR购买的容量已用完或者签名错误")
    elif ocr_str_conNum == 500:
        msg_conNum = "阿里OCR服务器错误，请稍后重试"
        print("箱号识别：阿里OCR服务器错误，请稍后重试")
    elif ocr_str_conNum == 600:
        msg_conNum = "无法识别该图片，请重新上传图片"
        print("箱号识别：无法识别该图片，请重新上传图片")
    elif ocr_str_conNum == 700:
        msg_conNum = "解析出错，请重新上传图片"
        print("箱号识别：解析出错，请重新上传图片")
    elif type(ocr_str_conNum) == int:
        msg_conNum = "网络错误，请稍后再试"
        print("箱号识别：网络错误，请稍后再试")
    elif not ocr_str_conNum:
        msg_conNum = "无法识别该图片，请重新上传图片"
        print("箱号识别：无法识别该图片，请重新上传图片")
    elif ocr_str_conNum == "{" or ocr_str_conNum == "}" or ocr_str_conNum == "\"":
        msg_conNum = "无法识别该图片，请重新上传图片"
        print("箱号识别：无法识别该图片，请重新上传图片")
    else:
        res_conNum = containerNum.extractWordsFromOCRResult(ocr_str_conNum)
        print(res_conNum)
        if res_conNum == 700:
            res_conNum = ""
            msg_conNum = "解析出错，请重新上传图片"
            print("箱号识别：解析出错，请重新上传图片")
            # return json.dumps({"containerNum": res, "msg_conNum": msg_conNum, "success": success})
        else:
            res_conNum = containerNum.rectifyResult(res_conNum)
            print(res_conNum)
            if res_conNum == 700:
                res_conNum = ""
                msg_conNum = "解析出错，请重新上传图片"
                print("箱号识别：解析出错，请重新上传图片")
                # return json.dumps({"containerNum": res, "msg_conNum": msg_conNum, "success": success})
            else:
                success_conNum = 1

    if ocr_str_weight == 400:
        msg_weight = "阿里OCR参数错误"
        print("重量识别：阿里OCR参数错误")
    elif ocr_str_weight == 401:
        msg_weight = "您无阿里OCR的权限，请开通后使用"
        print("重量识别：您无阿里OCR的权限，请开通后使用")
    elif ocr_str_weight == 403:
        msg_weight = "阿里OCR购买的容量已用完或者签名错误"
        print("重量识别：阿里OCR购买的容量已用完或者签名错误")
    elif ocr_str_weight == 500:
        msg_weight = "阿里OCR服务器错误，请稍后重试"
        print("重量识别：阿里OCR服务器错误，请稍后重试")
    elif ocr_str_weight == 600:
        msg_weight = "无法识别该图片，请重新上传图片"
        print("重量识别：无法识别该图片，请重新上传图片")
    elif ocr_str_weight == 700:
        msg_weight = "重量解析出错，请重新上传图片"
        print("重量识别：重量解析出错，请重新上传图片")
    elif type(ocr_str_weight) == int:
        msg_weight = "网络错误，请稍后再试"
        print("重量识别：网络错误，请稍后再试")
    elif not ocr_str_weight:
        msg_weight = "无法识别该图片，请重新上传图片"
        print("重量识别：无法识别该图片，请重新上传图片")
    elif ocr_str_weight == "{" or ocr_str_weight == "}" or ocr_str_weight == "\"":
        msg_weight = "无法识别该图片，请重新上传图片"
        print("重量识别：无法识别该图片，请重新上传图片")
    else:
        res_weight = containerNum.extractWeightOCR(ocr_str_weight)

        if res_weight == 700 or res_weight == ["", ""]:
            msg_weight = "重量解析出错，请重新上传图片"
            print("重量识别：重量解析出错，请重新上传图片")
            # return json.dumps({"containerNum": res, "msg_conNum": msg_conNum, "success": success})
        else:
            success_weight = 1
            res_weight_kg = res_weight[0]
            res_weight_lb = res_weight[1]

    return json.dumps({"res": res_conNum, "msg": msg_conNum, "success": success_conNum, "weight_kg": res_weight_kg, "weight_lb": res_weight_lb, "msg_weight": msg_weight, "success_weight": success_weight})

#提取铅封号的route.
@app.route('/qianfenghao', methods=['get', 'post'])
def mashiji():

    if 'path' in request.args:
        infos = request.args['path']
    else:
        tmp = parser_put.parse_args()
        infos = tmp['path']

    url_request = "https://tysbgpu.market.alicloudapi.com/api/predict/ocr_general"
    appcode = '4335919ca03a4714832942f64c6cef2b'

    mashiji_Num = API_mashiji(image_dir=infos, appcode=appcode, url_request=url_request)

#    encodestr = mashiji_Num.imageToOCR()
    encodestr = mashiji_Num.urlImage()
    encodestr = {'image': encodestr}
    ocr_str = mashiji_Num.posturl(data=encodestr)
#    print(ocr_str)

    success = 0
    msg = ""
    res = ""
    if ocr_str == 400:
        msg = "阿里OCR参数错误"
        print("铅封号识别：阿里OCR参数错误")
    elif ocr_str == 401:
        msg = "您无阿里OCR的权限，请开通后使用"
        print("铅封号识别：您无阿里OCR的权限，请开通后使用")
    elif ocr_str == 403:
        msg = "阿里OCR购买的容量已用完或者签名错误"
        print("铅封号识别：阿里OCR购买的容量已用完或者签名错误")
    elif ocr_str == 500:
        msg = "阿里OCR服务器错误，请稍后重试"
        print("铅封号识别：阿里OCR服务器错误，请稍后重试")
    elif ocr_str == 600:
        msg = "无法识别该图片，请重新上传图片"
        print("铅封号识别：无法识别该图片，请重新上传图片")
    elif not ocr_str:
        msg = "无法识别该图片，请重新上传图片"
        print("铅封号识别：无法识别该图片，请重新上传图片")
    elif type(ocr_str) == int:
        msg = "网络错误，请稍后再试"
        print("铅封号识别：网络错误，请稍后再试")
    elif not ocr_str:
        msg = "无法识别该图片，请重新上传图片"
        print("铅封号识别：无法识别该图片，请重新上传图片")
    elif ocr_str == "{" or ocr_str == "}" or ocr_str == "\"":
        msg = "无法识别该图片，请重新上传图片"
        print("铅封号识别：无法识别该图片，请重新上传图片")
    else:
        res = mashiji_Num.extractWordsFromOCRResult(ocr_str)
        if res == 700:
            res = ""
            msg = "解析出错，请重新上传图片"
            print("铅封号识别：解析出错，请重新上传图片")
            return json.dumps({"res": res, "msg": msg, "success": success})
        else:
            success = 1

    return json.dumps({"res": res, "msg": msg, "success": success})

if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0', port=8222)
