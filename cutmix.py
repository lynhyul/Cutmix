from keras.models import Sequential
#Import from keras_preprocessing not from keras.preprocessing
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from keras.optimizers import RMSprop
import pandas as pd
import numpy as np


import random
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import PIL
from PIL import ImageDraw
from keras import regularizers
from keras import models, layers, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import imgaug as ia
from imgaug import augmenters as iaa
from keras.applications.mobilenet import preprocess_input


def append_ext(fn):
    return fn+".png"
traindf=pd.read_csv('C:/data/image/cifar10/trainLabels.csv',dtype=str)
testdf=pd.read_csv("C:/data/image/cifar10/sampleSubmission.csv",dtype=str)
traindf["id"]=traindf["id"].apply(append_ext)
testdf["id"]=testdf["id"].apply(append_ext)
datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)


train_generator1=datagen.flow_from_dataframe(
dataframe=traindf,
directory="C:/data/image/cifar10/train/train/",
x_col="id",
color_mode='rgb',
y_col="label",
subset="training",
batch_size=32,
shuffle=True,
class_mode="categorical",
target_size=(32,32))


train_generator2=datagen.flow_from_dataframe(
dataframe=traindf,
directory="C:/data/image/cifar10/train/train/",
x_col="id",
y_col="label",
color_mode='rgb',
subset="training",
batch_size=32,
shuffle=True,
class_mode="categorical",
target_size=(32,32))

valid_generator=datagen.flow_from_dataframe(
dataframe=traindf,
directory="C:/data/image/cifar10/train/train/",
x_col="id",
y_col="label",
color_mode='rgb',
subset="validation",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(32,32))

test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_dataframe(
dataframe=testdf,
directory="C:/data/image/cifar10/test/test/",
x_col="id",
color_mode='rgb',
y_col=None,
batch_size=32,
seed=42,
shuffle=False,
class_mode=None,
target_size=(32,32))

from cutmix_keras import CutMixImageDataGenerator  # Import CutMix
import efficientnet.keras as efn 


from cutmix_keras import CutMixImageDataGenerator
train_generator = CutMixImageDataGenerator(
    generator1=train_generator1,
    generator2=train_generator2,
    img_size=32,
    batch_size=32,
)


from keras.layers import BatchNormalization
from keras.applications import MobileNet, ResNet50
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,LSTM, GlobalAveragePooling2D


mobile = MobileNet(weights='imagenet', include_top=False,input_shape=(32,32,3))

model = Sequential()
model.add(mobile)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(optimizers.Adam(lr=0.0001, decay=1e-5),loss="categorical_crossentropy",metrics=["accuracy"])

es = EarlyStopping(patience=20)
lr = ReduceLROnPlateau(patience=10,factor=0.5)

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_generator1.n//train_generator1.batch_size,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=200, callbacks= [es,lr]

)


model.evaluate_generator(generator=valid_generator,
steps=STEP_SIZE_TEST)


test_generator.reset()
pred=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)


predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]


sub = pd.read_csv('C:/data/image/cifar10/sampleSubmission.csv')
sub['label'] = predictions
sub.to_csv("C:/data/image/cifar10/results_cutmix.csv",index=False)




# no cutmix
# loss: 0.0481 - accuracy: 0.9843 - val_loss: 0.7864 - val_accuracy: 0.8235
# score : 82

# cutmix
# Epoch 41/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.1947 - accuracy: 0.7193 - val_loss: 0.5306 - val_accuracy: 0.8423
# Epoch 42/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.1894 - accuracy: 0.7240 - val_loss: 0.5252 - val_accuracy: 0.8485
# Epoch 43/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.1660 - accuracy: 0.7322 - val_loss: 0.5461 - val_accuracy: 0.8439
# Epoch 44/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.1738 - accuracy: 0.7323 - val_loss: 0.5242 - val_accuracy: 0.8469
# Epoch 45/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.1680 - accuracy: 0.7298 - val_loss: 0.5288 - val_accuracy: 0.8478
# Epoch 46/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.1602 - accuracy: 0.7357 - val_loss: 0.5264 - val_accuracy: 0.8460
# Epoch 47/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.1588 - accuracy: 0.7373 - val_loss: 0.5306 - val_accuracy: 0.8450
# Epoch 48/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.1589 - accuracy: 0.7389 - val_loss: 0.5353 - val_accuracy: 0.8470
# Epoch 49/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.1530 - accuracy: 0.7392 - val_loss: 0.5210 - val_accuracy: 0.8475
# Epoch 50/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.1570 - accuracy: 0.7382 - val_loss: 0.5335 - val_accuracy: 0.8438
# Epoch 51/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.1521 - accuracy: 0.7391 - val_loss: 0.5316 - val_accuracy: 0.8428
# Epoch 52/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.1393 - accuracy: 0.7415 - val_loss: 0.5142 - val_accuracy: 0.8482
# Epoch 53/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.1343 - accuracy: 0.7423 - val_loss: 0.5360 - val_accuracy: 0.8456
# Epoch 54/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.1366 - accuracy: 0.7408 - val_loss: 0.5237 - val_accuracy: 0.8506
# Epoch 55/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.1298 - accuracy: 0.7507 - val_loss: 0.5239 - val_accuracy: 0.8495
# Epoch 56/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.1328 - accuracy: 0.7478 - val_loss: 0.5293 - val_accuracy: 0.8477
# Epoch 57/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.1288 - accuracy: 0.7510 - val_loss: 0.5226 - val_accuracy: 0.8505
# Epoch 58/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.1224 - accuracy: 0.7513 - val_loss: 0.5266 - val_accuracy: 0.8492
# Epoch 59/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.1158 - accuracy: 0.7551 - val_loss: 0.5312 - val_accuracy: 0.8489
# Epoch 60/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.1161 - accuracy: 0.7520 - val_loss: 0.5313 - val_accuracy: 0.8481
# Epoch 61/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.1208 - accuracy: 0.7509 - val_loss: 0.5219 - val_accuracy: 0.8483
# Epoch 62/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.1105 - accuracy: 0.7538 - val_loss: 0.5302 - val_accuracy: 0.8472
# Epoch 63/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.1018 - accuracy: 0.7596 - val_loss: 0.5185 - val_accuracy: 0.8507
# Epoch 64/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.0921 - accuracy: 0.7652 - val_loss: 0.5159 - val_accuracy: 0.8511
# Epoch 65/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.0968 - accuracy: 0.7637 - val_loss: 0.5168 - val_accuracy: 0.8509
# Epoch 66/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.0838 - accuracy: 0.7739 - val_loss: 0.5173 - val_accuracy: 0.8527
# Epoch 67/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.0917 - accuracy: 0.7660 - val_loss: 0.5255 - val_accuracy: 0.8473
# Epoch 68/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.0901 - accuracy: 0.7680 - val_loss: 0.5230 - val_accuracy: 0.8492
# Epoch 69/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.0725 - accuracy: 0.7728 - val_loss: 0.5166 - val_accuracy: 0.8515
# Epoch 70/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.0810 - accuracy: 0.7720 - val_loss: 0.5226 - val_accuracy: 0.8478
# Epoch 71/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.0827 - accuracy: 0.7731 - val_loss: 0.5240 - val_accuracy: 0.8476
# Epoch 72/200
# 1171/1171 [==============================] - 20s 17ms/step - loss: 1.0733 - accuracy: 0.7747 - val_loss: 0.5173 - val_accuracy: 0.8478
# score : 
