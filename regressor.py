#线性回归
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense

X=np.linspace(-1,1,200)
np.random.shuffle(X)
Y=0.5*X+2+np.random.normal(0,0.05,(200,))
plt.scatter(X,Y)

X_train,Y_train=X[:150],Y[:150]
X_test,Y_test=X[150:],Y[150:]

#创建神经网络
model=Sequential()
model.add(Dense(units=1,input_dim=1))
#搭建神经网络
model.compile(loss='mse',optimizer="sgd")

print("Training ------")
for step in range(301) :
    cost=model.train_on_batch(X_train,Y_train)
    if step%100==0:
        print("train cost:",cost)

print('\nTesting---------')
cost=model.evaluate(X_test,Y_test,batch_size=50)
print('test cost:',cost)
W,b=model.layers[0].get_weights()
print("Weights=",W,"\nbiases=",b)

Y_pred=model.predict(X_test)
plt.scatter(X_test,Y_test)
plt.plot(X_test,Y_pred)
plt.show()
