from keras.models import load_model
import tensorflow as tf
import json
import numpy as np
import cv2
# from wavelet import w2d

__class_name_to_number = {}
__class_number_to_name = {}
__model = None

def classify_image(file_path=None):

    img = cv2.imread(file_path)
    result = []
    resize = tf.image.resize(img, (256, 256))
    result.append({
            # 'class': class_number_to_name(__model.predict_classes(final)[0]),
            'class_probability': np.around(__model.predict(np.expand_dims(resize/255, 0)),2).tolist()[0],
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


if __name__ == '__main__':
    load_saved_artifacts()
    print(classify_image(r"E:\Project\GeorgeClassification\Test images\hunter.jpeg"))
