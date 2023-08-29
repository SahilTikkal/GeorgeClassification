import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.metrics import Precision, Recall, BinaryAccuracy

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Remove dodgy images
# image_exts = ['jpeg','jpg', 'bmp', 'png']
# data_dir = 'data'
# # for image_class in os.listdir(data_dir):
# #     for image in os.listdir(os.path.join(data_dir, image_class)):
# #         image_path = os.path.join(data_dir, image_class, image)
# #         try:
# #             img = cv2.imread(image_path)
# #             tip = imghdr.what(image_path)
# #             if tip not in image_exts:
# #                 print('Image not in ext list {}'.format(image_path))
# #                 os.remove(image_path)
# #         except Exception as e:
# #             print('Issue with image {}'.format(image_path))
# #             os.remove(image_path)

# Load Data

# creating dataset using keras api
data = tf.keras.utils.image_dataset_from_directory(r'C:\Users\SAHIL\PycharmProjects\Data_Engineer_Test\data')

data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
# plotting samples
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

# Scale Data
data = data.map(lambda x,y: (x/255, y))

# Split Data
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

# Build Deep Learning Model
model = Sequential()  # model initialization
# Hidden layers
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# output layer
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

# Train
logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=50, validation_data=val, callbacks=[tensorboard_callback])


pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f"{pre.result()}, {re.result()}, {acc.result()}")


model.save(os.path.join('models','imageclassifier.h5'))