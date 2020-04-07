import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as processimage

model = load_model('my_model.h5')

class MainPredictImg(object):
    def __init__(self):
        pass
    def pred(self,filename):
        #读取照片
        pred_img = processimage.imread(filename)
        #转换成np array
        pred_img = np.array(pred_img)
        #转换成神经网络可以的结构
        pred_img = pred_img.reshape(-1,28,28,1)
        #predict
        prediction = model.predict(pred_img)
        Final_prediction = [result.argmax() for result in prediction][0]
        a=0
        for i in prediction[0]:
            print(a)
            print('Percent:{:.3%}'.format(i))
            a=a+1
        #format
        print(a)
        return Final_prediction

def main():
    Predict = MainPredictImg()
    res = Predict.pred('3.jpg')
    print ("your number is:-->",res)

if __name__ == '__main__':
    main()
