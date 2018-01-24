# simple linear regression with 1 variable implementation in TensorFlow

import tensorflow as tf
import numpy
import numpy.random as random
import matplotlib.pyplot as plt
#parameters
l_r = 0.01
epochs = 100
disp_step = 50

# Training Data
trainX = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
trainY = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = trainX.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(random.randn(),name="weight")
b = tf.Variable(random.randn(),name="bias")
#lin_reg = W*X+b
pred = tf.add(tf.multiply(X,W),b)

# loss function
cost = tf.reduce_sum(tf.pow(Y-pred,2))/(2*n_samples)
# Gradient Descent
optimizer  = tf.train.GradientDescentOptimizer(l_r).minimize(cost)

# initialize global variables
init = tf.global_variables_initializer()

# start training session
with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(epochs):
        for (x,y) in zip(trainX,trainY):
            sess.run(optimizer,feed_dict={X:x,Y:y})
            
            #display logs per epoch step
            if (epoch+1)%disp_step == 0:
                c = sess.run(cost,feed_dict={X:trainX,Y:trainY})
                print("Epoch:",'%04d'%(epoch+1),"Cost:","{:.9f}".format(c),"W=",sess.run(W),"b=",sess.run(b),"\n")
                
    print ("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: trainX, Y: trainY})
    print ("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')     
    
    #Graphic display
    plt.plot(trainX,trainY,'ro',label='Original Data')
    plt.plot(trainX,sess.run(W)*trainX+sess.run(b),label='Fitted Line')
    plt.legend()
    plt.show()
