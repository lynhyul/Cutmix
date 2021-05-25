import numpy as np
from cutmix_keras import CutMixImageDataGenerator  # Import CutMix
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, Flatten,BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.layers import UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from keras.datasets import cifar100
from keras.applications.mobilenet import preprocess_input
from keras.optimizers import Adam

import numpy as np



(x_train, y_train), (x_test,y_test)= cifar100.load_data()

from matplotlib import pyplot as plt

plt.imshow(x_train[1], cmap='gray')
plt.show()


# # 데이터 전처리

x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

print(x_train.shape)    # (50000, 32, 32, 3)



from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


idg = ImageDataGenerator(
    # rotation_range=10, acc 하락
    width_shift_range=(-1,1),  
    height_shift_range=(-1,1), 
    rotation_range=45, 
    # shear_range=0.2)    # 현상유지
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

idg2 = ImageDataGenerator()

train_generator1 = idg.flow(x_train,y_train,batch_size=64, seed = 2048)
# seed => random_state
valid_generator = idg2.flow(x_test,y_test)           

train_generator = CutMixImageDataGenerator(
    generator1=train_generator1,
    generator2=valid_generator,
    img_size=32,
    batch_size=64,
)


# mobile = MobileNet(weights='imagenet', include_top=False,input_shape=(32,32,3))



# mobile.trainable = True


# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,LSTM
# model = Sequential()
# # model.add(UpSampling2D(size=(3,3)))
# model.add(mobile)
# model.add(GlobalAveragePooling2D())
# model.add(Dense(100, activation= 'softmax'))

# mobile.summary()
# for i, layer in enumerate(mobile.layers):
#        print(i, layer.name)

# '''
# 0 input_1
# 1 block1_conv1
# 2 block1_conv1_bn
# 3 block1_conv1_act
# 4 block1_conv2
# 5 block1_conv2_bn
# 6 block1_conv2_act
# 7 block2_sepconv1
# ...
# 125 add_11
# 126 block14_sepconv1
# 127 block14_sepconv1_bn
# 128 block14_sepconv1_act
# 129 block14_sepconv2
# 130 block14_sepconv2_bn
# 131 block14_sepconv2_act
# ====================================
# Total params: 20,861,480
# Trainable params: 0
# Non-trainable params: 20,861,480
# '''



# # print("그냥 가중치의 수 : ", len(model.weights))   #32 -> (weight가 있는 layer * (i(input)bias + o(output)bias))
# # print("동결 후 훈련되는 가중치의 수 : ",len(model.trainable_weights))   #6
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
# modelpath = './modelCheckpoint/k46_cifa10_{epoch:02d}-{val_loss:.4f}.hdf5'
# early_stopping = EarlyStopping(monitor='val_loss', patience= 20)
# cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',save_best_only=True,mode='auto')
# lr = ReduceLROnPlateau(factor=0.1,verbose=1,patience=10)

# model.compile(loss = 'categorical_crossentropy', optimizer=Adam(1e-5), metrics=['acc'])
# history = model.fit_generator(train_generator,epochs=80, steps_per_epoch= len(x_train) / 64,
#     validation_data=valid_generator, callbacks=[early_stopping,lr])

# #4. evaluate , predict

# loss = model.evaluate(x_test,y_test, batch_size=1)
# print("loss : ",loss)


# y_predict = model.predict(x_test)
# y_Mpred = np.argmax(y_predict,axis=-1)
# print("y_test : ",y_test[:10])
# print("y_test : ",y_test[:10])

# import matplotlib.pyplot as plt
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.plot(history.history['acc'])       #회귀모델이기때문에 acc측정이 힘들다
# plt.plot(history.history['val_acc'])
# plt.title('loss & acc')
# plt.ylabel('loss & acc')
# plt.xlabel('epoch')
# plt.legend(['tran loss', 'val loss', 'train acc','val acc'])    #주석
# plt.show()

# # loss :  [1.9762324094772339, 0.4763999879360199]