import numpy as np
from keras.preprocessing.image import  array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave,imshow
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Activation, Dense, Dropout, Flatten, InputLayer,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
img_cols = 128
img_rows = 128
input_shape = (img_rows, img_cols, 1)
model = Sequential()
model.add(Conv2D(16, (5, 5),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (4, 4),activation='relu',input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(32, (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (4, 4), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(32768, activation='linear'))
model.compile(loss='mean_squared_error',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.load_weights('weights.hdf5')

def predictImage(img,name):
    t = img_to_array(img)
    t =rgb2lab(1.0/255*t)[:,:,0]
    t = t.reshape(1,128,128,1)
    out = model.predict(t)
    out = out.reshape(1,128,128,2)
    out *= 128
    cur = np.zeros((128, 128, 3))
    cur[:,:,0] = t[0][:,:,0]
    cur[:,:,1:] = out[0]
    imsave(name + ".jpg",lab2rgb(cur))
    imsave(name + "_bw.jpg", rgb2gray(lab2rgb(cur)))

im = load_img("playground/1184_128.jpg")

predictImage(im,"static/result")



