from keras.models import load_model
import tensorflow as tf
import json
import numpy as np
import base64
import cv2
# from wavelet import w2d



__class_name_to_number = {}
__class_number_to_name = {}
__model = None


def classify_image(image_base64_data, file_path=None):
    img = get_base64_image(file_path, image_base64_data)
    result = []
    resize = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    pred = round(__model.predict(np.expand_dims(resize / 255, 0)).tolist()[0][0], 2)
    # class_probability = [__model.predict(np.expand_dims(resize / 255, 0)),
    #               1-__model.predict(np.expand_dims(resize / 255, 0))]
    class_probability = [1-pred, pred]


    class_num = class_probability.index(max(class_probability))
    result.append({
            'class': class_number_to_name(class_num),
            'class_probability': class_probability,
            'class_dictionary': __class_name_to_number
        })
    return result

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open("./artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        __model = load_model(r'E:\Project\GeorgeClassification\server\artifacts\imageclassifier.h5')
    print("loading saved artifacts...done")


def get_base64_image(image_path, image_base64_data):
    if image_path is not None:
        img = cv2.imread(image_path)
        print(1)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)
    return img


def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_b64_test_image():
    with open(r"E:\Project\GeorgeClassification\Test images\st.txt", "r") as f:
        return f.read()


if __name__ == '__main__':
    load_saved_artifacts()
    # print(classify_image(get_b64_test_image(), None))
    print(classify_image(None,r"E:\Project\GeorgeClassification\Test images\dragon.jpg"))
