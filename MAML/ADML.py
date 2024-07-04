import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def sample_points(k):
    x = np.random.rand(k,50)
    y = np.random.choice([0, 1], size=k, p=[.5, .5]).reshape([-1,1])
    return x,y

x, y = sample_points(10)
print(x[0])
print(y[0])

def FGSM(x,y):

    #placeholder for the inputs x and y
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    #initialize theta with random values
    theta = tf.Variable(tf.zeros([50,1]))

    #predict the value of y
    YHat = tf.nn.softmax(tf.matmul(X, theta)) 

    #calculate the loss
    loss = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(YHat), reduction_indices=1))
 
    #now calculate gradient of our loss function with respect to our input X instead of model parameter theta
    gradient = ((tf.gradients(loss,X)[0]))
    
    #calculate the adversarial input
    #i.e x_adv = x + epsilon * sign ( nabla_x J(X, Y))
    X_adv = X + 0.2*tf.sign(gradient)
    X_adv = tf.clip_by_value(X_adv,-1.0,1.0)

    #start the tensoflow session
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())        
        X_adv = sess.run(X_adv, feed_dict={X: x, Y: y})
        
    return X_adv, y

X_adv, y = FGSM(x,y)

class ADML(object):
    def __init__(self):

        #initialize number of tasks i.e number of tasks we need in each batch of tasks
        self.num_tasks = 2
        
        #number of samples i.e number of shots  -number of data points (k) we need to have in each task
        self.num_samples = 10

        #number of epochs i.e training iterations
        self.epochs = 100
        
        #hyperparameter for the inner loop (inner gradient update)

        #for clean sample
        self.alpha1 = 0.0001

        #for adversarial sample
        self.alpha2 = 0.0001
        
        #hyperparameter for the outer loop (outer gradient update) i.e meta optimization
        
        #for clean sample
        self.beta1 = 0.0001
        
        #for adversarial sample
        self.beta2 = 0.0001

        #randomly initialize our model parameter theta
        self.theta = np.random.normal(size=50).reshape(50, 1)

    #define our sigmoid activation function  
    def sigmoid(self,a):
        return 1.0 / (1 + np.exp(-a))
    
    #now let us get to the interesting part i.e training :P
    def train(self):
        
        #for the number of epochs,
        for e in range(self.epochs):        
            
            #theta' of clean samples
            self.theta_clean = []

            #theta' of adversarial samples
            self.theta_adv = []
            
            #for task i in batch of tasks
            for i in range(self.num_tasks):
                
                #sample k data points and prepare our training data

                #first, we sample clean data points
                XTrain_clean, YTrain_clean = sample_points(self.num_samples)

                #feed the clean samples to FGSM and get adversarial samples
                XTrain_adv, YTrain_adv = FGSM(XTrain_clean,YTrain_clean)

                #1. First, we computer theta' for clean samples and store it in theta_clean
                
                #predict the output y 
                a = np.matmul(XTrain_clean, self.theta)

                YHat = self.sigmoid(a)

                #since we are performing classification, we use cross entropy loss as our loss function
                loss = ((np.matmul(-YTrain_clean.T, np.log(YHat)) - np.matmul((1 -YTrain_clean.T), np.log(1 - YHat)))/self.num_samples)[0][0]
                
                #minimize the loss by calculating gradients
                gradient = np.matmul(XTrain_clean.T, (YHat - YTrain_clean)) / self.num_samples

                #update the gradients and find the optimal parameter theta' for clean samples
                self.theta_clean.append(self.theta - self.alpha1*gradient)
              
                #2. Now, we compute theta' for adversarial samples and store it in theta_clean

                #predict the output y 
                a = (np.matmul(XTrain_adv, self.theta))

                YHat = self.sigmoid(a)

                #calculate cross entropy loss
                loss = ((np.matmul(-YTrain_adv.T, np.log(YHat)) - np.matmul((1 -YTrain_adv.T), np.log(1 - YHat)))/self.num_samples)[0][0]
                
                #minimize the loss by calculating gradients
                gradient = np.matmul(XTrain_adv.T, (YHat - YTrain_adv)) / self.num_samples

                #update the gradients and find the optimal parameter theta' for adversarial samples
                self.theta_adv.append(self.theta - self.alpha2*gradient)
                
            #initialize meta gradients for clean samples
            meta_gradient_clean = np.zeros(self.theta.shape)

            #initialize meta gradients for adversarial samples
            meta_gradient_adv = np.zeros(self.theta.shape)
            
            for i in range(self.num_tasks):
                
                #sample k data points and prepare our test set for meta training

                #first, we sample clean data points
                XTest_clean, YTest_clean = sample_points(self.num_samples)

                #feed the clean samples to FGSM and get adversarial samples
                XTest_adv, YTest_adv = FGSM(XTest_clean,YTest_clean)
                       
                #1. First, we computer meta gradients for clean samples 

                #predict the value of y
                a = np.matmul(XTest_clean, self.theta_clean[i])
                
                YPred = self.sigmoid(a)
                           
                #compute meta gradients
                meta_gradient_clean += np.matmul(XTest_clean.T, (YPred - YTest_clean)) / self.num_samples

                #2. Now, we compute meta gradients for adversarial samples
                
                #predict the value of y
                a = (np.matmul(XTest_adv, self.theta_adv[i]))
                
                YPred = self.sigmoid(a)
                           
                #compute meta gradients
                meta_gradient_adv += np.matmul(XTest_adv.T, (YPred - YTest_adv)) / self.num_samples

            #update our randomly initialized model parameter theta
            #with the meta gradients of both clean and adversarial samples
            
            self.theta = self.theta-self.beta1*meta_gradient_clean/self.num_tasks

            self.theta = self.theta-self.beta2*meta_gradient_adv/self.num_tasks
                                  
            if e%10==0:
                print("Epoch {}: Loss {}\n".format(e,loss))             
                print('Updated Model Parameter Theta\n')
                print('Sampling Next Batch of Tasks \n')
                print('---------------------------------\n')

model = ADML()
model.train()
print("Training Completed!")