#卷积神经网络：对图片的每一块像素区域进行处理
import numpy as np
from keras.datasets import mnist
#处理numpy数据
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D,Activation,MaxPool2D,Flatten,Dense
from keras.optimizers import Adam

nb_class = 10
#训练的次数
nb_epoch = 2
#每次训练给神经网络多少数据
batchsize = 128

#准备数据集
(X_train,Y_train),(X_test,Y_test) = mnist.load_data()

#setup data shape
X_train = X_train.reshape(-1,28,28,1)
# 灰度照片只有1个维度，彩色RGB：3个维度
#-1是未知
X_test = X_test.reshape(-1,28,28,1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#形式上是浮点 需要归一化：真正变成小数
X_train = X_train/255
X_test = X_test/255
#One-hot [0,0,0,0,0,1,0,0,0]=5
Y_train = np_utils.to_categorical(Y_train,nb_class)
Y_test = np_utils.to_categorical(Y_test,nb_class)

model = Sequential()
#1st Conv2D layer
#卷积（长宽减小，高度增加，像素缩减）
#将压缩的信息放到普通的分类神经层上就可以分类
#每次卷积都会遗失部分信息，所以引入了池化层
#用池化压缩后然后两层全连接层，然后分类预测
#两层卷积池化加两层全连接 然后分类
model.add(Convolution2D(
    filters = 32,
    #图上放32个过滤器
    kernel_size =(5,5),
    #5个像素（一个过滤器过滤5*5的尺寸）
    padding = 'same',
    input_shape = (28,28,1)
))
model.add(Activation('relu'))
model.add(MaxPool2D(
    pool_size = (2,2),
    #抓取2*2
    strides = (2,2),
    #步长2*2
    padding = 'same',
))
#2nd Conv2D layer
model.add(Convolution2D(
    filters = 64,
    kernel_size = (5,5),
    padding = 'same',
))
model.add(Activation('relu'))
model.add(MaxPool2D(
    pool_size =(2,2),
    strides = (2,2),
    padding = "same",
))
#[[[1,2,3,4],[1,2,3,4],[1,2,3,4]]]
#flatten都压缩了
#1st Fully connected Dense
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

#2nd Fully connected Dense
model.add(Dense(nb_class))
model.add(Activation('softmax'))

#Define Optimizer and setup Param
adam=Adam(lr=0.0001) #太大容易过拟合
#compile model
model.compile(optimizer=adam,loss='categorical_crossentropy',
              metrics=['accuracy'],
              )
#Run/Fireup network
model.fit(x=X_train,
          y=Y_train,
          epochs=nb_epoch,
          batch_size=batchsize,
          verbose=1,
          validation_data=(X_test,Y_test),
          )
model.save('my_model.h5')
#loss: 0.0527 - acc: 0.9838 - val_loss: 0.0387 - val_acc: 0.9867
#2次  loss: 0.0945 - acc: 0.9720 - val_loss: 0.0646 - val_acc: 0.9798