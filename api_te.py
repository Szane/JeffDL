import requests
import json


# api路径
url = "http://127.0.0.1:8222/recognition"

parms = {
    'path': './test_image/image001.jpg',  # 发送给服务器的内容
#    'path': 'test.png'
}

# headers = {
#     'User-agent': 'none/ofyourbusiness',
#     'Spam': 'Eggs'
# }

res = requests.post(url, data=parms)  # 发送请求

text = res.text
print(text)

