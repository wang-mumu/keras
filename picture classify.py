#图片分类
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import  Sequential
import matplotlib.image as precessimage
import matplotlib.pyplot as plt
from keras.layers.core import Dense,Activation,Dropout
from keras.optimizers import RMSprop

#拉取数据集
(X_train,Y_train),(X_test,Y_test)=mnist.load_data()
#X shape(60000,28*28),y shape(10000,)y对应图片类别

#图片数据需要准备成神经网络可以接受的维度
#准备数据 全连接层只能认识单行
#reshape
X_train=X_train.reshape(60000,784)
X_test=X_test.reshape(10000,784)
#把数据设置成小数更好
#astype 转换类型
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
#形式上是浮点 需要归一化：真正变成小数
X_train=X_train/255
X_test/=255

#准备基本参数
batch_size=1024 #每次给神经网络多少数据
nb_class=10  #图片分类分成10个类
nb_epochs=7 #训练的次数
#训练集有1000个样本，batch_size=10,训练完整个样本集需要：100次迭代，1次epoch
#epochs指的就是训练过程中数据将被“轮”多少次

#Class vectors [0,0,0,0,0,0,0,1(7),0,0] 标签
#3：0 0 0 1
#将整数的类别标签转成onehot编码
Y_test=np_utils.to_categorical(Y_test,nb_class)
Y_train=np_utils.to_categorical(Y_train,nb_class)

#设置网络结构
model=Sequential()
#第一层网络
model.add(Dense(512,input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

#2nd layer
model.add(Dense(256))  #第二层只需要告诉输出，输入系统自动识别
model.add(Activation('relu'))
model.add(Dropout(0.2))

#3nd layer
model.add(Dense(10))
model.add(Activation('softmax'))
#softmax针对10个进行分类

#编译compile
model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
#adam,SGD不同路径进行的优化
    metrics=['accuracy'],)
#矩阵里达到accuracy的效果

#启动网络训练
Traning=model.fit(X_train,Y_train,batch_size=batch_size,
                  epochs=nb_epochs,validation_data=(X_test,Y_test),
                  verbose=2,)

#拉取test中第9998里的图做测试
testrun=X_test[9998].reshape(1,784)
testlabel=Y_test[9998]
print('label:-->>',testlabel)
plt.imshow(testrun.reshape([28,28]))
plt.show()

#判定输出结果
pred=model.predict(testrun)
print(pred)
print(pred.argmax())
