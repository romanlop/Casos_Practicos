# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 12:18:51 2019

@author: rlopseo
"""

import theano
import math
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt


trX = np.linspace(-1,1,101)
trY = np.linspace(-1,1,101)
for i in range(len(trY)):
    trY[i] = math.log(1 + 0.5 * abs(trX[i])) / 3 + np.random.rand() * 0.033
    
X = T.scalar()
Y = T.scalar()

def model(X, w):
    return T.log(1 + w[0] * T.abs_(X)) + w[1] * X

w = theano.shared(np.array([1,1], dtype = theano.config.floatX))
y = model(X, w)
 
cost = T.mean(T.sqr(y - Y))
gradient = T.grad(cost=cost, wrt=w)
update = [[w, w - gradient * 0.01]]
 
train = theano.function(inputs=[X, Y], outputs=cost, updates=update,
                        allow_input_downcast=True)
 
for i in range(100):
    for x, y in zip(trX, trY):
        train(x, y)
 
print(w.get_value())  
pesos = w.get_value()

result = []
for i in trX:
    result.append(math.log(1 + pesos[0] * abs(i)) + pesos[1] * i)


plt.plot(trX, trY, 'g.')
plt.plot(trX, result)
plt.show()