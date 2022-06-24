######################################################
# Created by April Nellis, advised by Dr. Erhan Bayraktar and Dr. Asaf Cohen
# Original model: Carmona, Ludkovski (2008), Example 1
# Altered to allow up higher-dimensional examples with same fuel dynamics
# No implementation delay
# Trains each mode individually
# Learns Y, Z, and Delta Y (jumps in value function)
######################################################
import numpy as np
import tensorflow as tf
import scipy
import time
from datetime import datetime
from tensorflow import keras
import warnings
import visualsML
import matplotlib.pyplot as plt
import math

# Just disables the annoying warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

warnings.simplefilter("error", RuntimeWarning)

''' --------------- MODEL PARAMETERS ---------------- '''
#example = 1 # CL 2005 example
example = 2 # a made-up 10-dimensional example to test speed etc.

T = 1/4 #3 months = 1/4 year
N = 180 #number of time slices, N+1 points
dt = T/N #1/4 of a day, 1/1440 of a year
timing = np.zeros(N)

if example == 1:
    d = 2 #[price of power, price of gas]
    x0 = np.array([[50.0, 6.0]]) #P_0 = 50, G_0 = 6
    lambda_poisson = 8 #average number of jumps in electricity prices per year
    lambda_exp = 10 #inverse of average intensity of jumps

    kappa = np.array([[5, 2]]) # 1 x d some kind of scaling coefficient
    mu = np.array([[math.log(50), math.log(6)]]) # 1 x d mean reversion parameters
    sigma = np.array([[0.5, 0.32], [0, 0.24]]) # d x d correlation for BM, sigma[i,j] corresponds to coefficient of ith BM for jth state variable

elif example == 2:
    d = 80
    x0 = np.ones(d)*6.0
    x0[0] = 50.0

    #np.array([[50.0, 6.0, 3.0, 10.0, 7.0, 8.0, 4.0, 5.0, 6.0, 5.0]]) # idk whatever, like 10 random numbers
    lambda_poisson = 8 #average number of jumps in electricity prices per year
    lambda_exp = 10 #inverse of average intensity of jumps

    #kappa = np.array([[5, 2, 2,  2,  2, 2,  2,  2,  2,  2]])
    kappa = np.ones(d) * 2.0
    kappa[0] = 5.0
    mu = np.log(x0) # let's keep the mean reversion simple lol

    #sigma = np.diag([ 0.29,  0.47,  0.24,  0.67,  0.73,  0.26,  0.14,  0.88,  0.37, 0.52]) # d x d correlation matrix, fuel prices uncorrelated with each other
    #sigma[0] = np.array([ 0.76,  0.55,  0.37,  0.16,  0.55,  0.63,  0.47,  0.39,  0.13, 0.47]) # electricity price correlated with all fuel prices

    sigma  = np.diag(np.ones(d)*0.24)
    sigma[0] = np.ones(d)*0.32
    sigma[0,0] = 0.5
    '''
    np.array([[ 0.76,  0.55,  0.37,  0.16,  0.55,  0.63,  0.47,  0.39,  0.13, 0.47],
       [ 0.29,  0.47,  0.24,  0.67,  0.73,  0.26,  0.14,  0.88,  0.37, 0.52],
       [ 0.13,  0.,  0.15,  0.,  0.14,  0.,  0.52,  0.73,  0.64, 0.61],
       [ 0.05,  0.62,  0.47,  0.42,  0.12,  0.,  0.96,  0.5 ,  0.58, 0.37],
       [ 0.,  0.95,  0.4 ,  0.84,  0.45,  0.,  0.66,  0.,  0.7 , 0.],
       [ 0.7 ,  0.,  0.89,  0.61,  1.09,  0.75,  0.38,  0.55,  0.56, 0.63],
       [ 0.74,  0.73,  0.,  0.28,  0.46,  0.59,  0.45,  1.02,  0.36, 1.09],
       [ 0.28,  0.96,  0.53,  0.51,  0.6 ,  0.44,  0.12,  0.27,  0.71, 1.  ],
       [ 0.,  0.,  0.,  0.06,  0.59,  0.3 ,  0.94,  0.85,  0.43, 0.32],
       [ 0.39,  0.41,  0.23,  0.63,  0.7 ,  0.37,  0.72,  0.34,  0.35, 0.36]])
     '''

else:
    print("uh oh, example not recognized")

capacity = np.array([0, 0.438, 0.876]) # production capacity in mode j
heat_rate = np.array([0, 7.5, 10]) # d-1 x J array of heat rates
prop_cap = np.array([1]) # d-1 sized array of proportion of capacity distributed to each fuel, must sum to 1

mat = np.array([[0, 0.438, .876],[0, -0.438*7.5, -0.876*10]])

#mat = np.zeros((d, J))
#mat[0] = capacity
#mat[1:] = -capacity * heat_rate * prop_cap

b = np.array([-1, -1.1, -1.2]) # J-sized array of fixed costs

J = 3 #number of operating regimes 0, ..., J-1

''' --------------- FLAGS ---------------- '''
normFlag = False # whether to normalize data
jumpFlag = True
trainFlag = True # whether to train or to used saved weights
animateFlag =  False # whether to generate an animation from visualizations
figFlag = True
earlyStopFlag = False
preInitFlag = False
if preInitFlag and example == 1:
    X0 = np.load(f'x_init.npy')[N]
else:
    X0 = np.expand_dims(x0, axis = 0)

''' --------------- NN STRUCTURE ---------------- '''
batchSize = 1000 #size of each minibatch
n_batches = 100 #number of batches
M_training = batchSize*n_batches # Size of the training set
myLearnRate = 1e-3
n_epochs = 101 #epoch = one pass through training data

#layer sizes
n_inputs = d # dimension of X
layerSize = d + 10
n_hidden1 = layerSize
n_hidden2 = layerSize
n_hidden3 = layerSize
n_hidden4 = layerSize
n_outputs = 1 + d + 1 # [Y, Z, U] where d is the shape of D_X(Y), U is Y(x + \xi) - Y(x)
n_layers = 4

class LossPrintingCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch%50 == 0:
            #file.write("The avg loss for epoch {} is {:.6f}, lr = {:.6g}. \n".format(epoch, logs["loss"], self.model.optimizer.lr.read_value()))
            print("The avg loss for epoch {} is {:.6f}, lr = {:.6g}.".format(epoch, logs["loss"], self.model.optimizer.lr.read_value()))


class LearningRateReducerCb(keras.callbacks.Callback):
    def on_train_begin(self, logs = None):
        if n >= (N - 20): self.model.optimizer.lr.assign(myLearnRate)
        elif n >= (N/2): self.model.optimizer.lr.assign(myLearnRate/10)
        else: self.model.optimizer.lr.assign(myLearnRate/100)

earlyStop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)

if earlyStopFlag:
    callback_list = [LossPrintingCallback(), LearningRateReducerCb(), earlyStop]
else:
    callback_list = [LossPrintingCallback(), LearningRateReducerCb()]


''' --------------- EXPERIMENT VARIABLES ---------------- '''
n_trials = 1 #how many times we train the model

# updated with each trial:
Y0_list = np.zeros((n_trials,J)) #stores the results of each trial

# overwritten each trial:
X_train = np.zeros((N+1, M_training, d)) #[time slice][sample in set][dimension of portfolio + jumps]

Y_hat = np.zeros((N+1, M_training, J + 1)) # final y and x -> [time slice][data entry][y^0, ..., y^{J-1}| X ]
Y_curr = np.zeros((M_training, J)) # y at current time step and mode BEFORE comparing to other modes
switches = np.ones((N+1, M_training, J), dtype = int) # optimal mode for each X at a given time
final = np.zeros(J)

''' --------------- FILE PATHS ---------------- '''

now = datetime.now()
dt_string = now.strftime("%m-%d-%H%M%S")

fig_path = f"Figures/{dt_string}" # file paths for figures to keep
anim_path = f"Animation/{dt_string}" # file path for images used in animation
weight_path = './Weights/weights2D-Z-' # file path for model weights

''' --------------- FUNCTIONS ---------------- '''

def generateData(aX):
    # initialize array
    half = aX.shape[1]//2
    rest = aX.shape[1] - half
    aX[0,:half] = x0 # first half are starting at initial point of interest
    if X0.shape[0] < half:
        rep = half//X0.shape[0]
        leftover = rep*X0.shape[0]
        aX[0,half:(half+leftover)] = np.tile(X0, (rep, 1))
        aX[0, (half+leftover):] = X0[0] #set remaining values at initial time slice equal to first element of X0
    else:
        aX[0,half:] = X0[:rest]

    #generate random paths
    for n in range(N):
        dW = np.random.normal(0,1,size=(M_training, d)) * np.sqrt(dt)
        step = kappa*(mu - np.log(aX[n]))*dt + np.matmul(dW, sigma)
        aX[n+1] = aX[n] * (1 + step)

        if jumpFlag: # apply jumps
            if example == 1:
                deltaN = np.random.binomial(1, lambda_poisson * dt, size = M_training) #Poisson process with rate jump_lambda * dt
                jump_size = np.random.exponential(1/lambda_exp, size = M_training)
                total_jump = np.exp(jump_size)**deltaN
                aX[n+1, :, 0] = aX[n+1, :, 0] * total_jump

            elif example == 2:
                # I choose some arbitrary jump configuration, whatever
                deltaN = np.random.binomial(1, lambda_poisson * dt, size = M_training) #Poisson process with rate jump_lambda * dt
                jump_size = np.random.exponential(1/lambda_exp, size = M_training)
                total_jump = np.exp(jump_size)**deltaN
                aX[n+1, :, 0] = aX[n+1, :, 0] * total_jump

            else:
                print('uh oh generateData')

    aX = np.float32(aX)

    return aX

def g_hat(X):
    return np.zeros_like(X)

def dg_hat(X):
    return np.zeros_like(X)

'''
# parameters for running cost/profit
if example == 1:
    #mat = np.array([[0, 0.438, .876],[0, -0.438*7.5, -0.876*10]])

    capacity = np.array([0, 0.438, 0.876]) # production capacity in mode j
    heat_rate = np.array([0, 7.5, 10]) # d-1 x J array of heat rates
    prop_cap = np.array([1]) # d-1 sized array of proportion of capacity distributed to each fuel, must sum to 1

elif example == 2:
    capacity = np.array([[0, 0.438, 0.876]]) # production capacity in mode j

    prop_cap = np.ones((d-1, J))*(1/d) # d-1 X J sized array of proportion of capacity distributed to each fuel in each mode, must sum to 1
    prop_cap[:,0] = 0 # when shut down in mode 0, no fuels contribute
    prop_cap[0,1] = 2/d # in mode 1 (half cap), the first fuel is most utilized
    prop_cap[d//2,2] = 2/d # in mode 2 (full), the d//2 fuel is most utilized
    print(prop_cap)

    if np.sum(prop_cap) != (J-1): #col must sum to 1
        print(f'sum is {np.sum(prop_cap)}, bad distribution of resources!')

    heat_rate = np.array([, # d-1 x J array of heat rates
                        [ 0.,  7,  9],
                        [ 0.,  8.5,  12],
                        [ 0.,  8,  11],
                        [ 0.,  9,  8],
                        [ 0.,  8.5,  13],
                        [ 0.,  6.5,  9.5],
                        [ 0.,  7,  11],
                        [ 0.,  5.5, 6.5]])
else:
    print("uh oh in f")
'''

def f_tensor(t, x, y):
    global mat, b
    mat_tf = tf.convert_to_tensor(mat, dtype = tf.float32)
    b_tf = tf.convert_to_tensor(b, dtype = tf.float32)
    profit_array = tf.linalg.matmul(x, mat_tf) + b_tf

    return profit_array #want to return tensor of shape (M_training, J)

def c(i,j,n):
    #return 0.05
    if example == 1:
        cost = 0.01*X_train[n,:,1]
    elif example == 2:
        cost = np.mean(0.01*X_train[n,:,1:], axis = 1)
    else:
        print("uh oh at switching cost")

    return cost #0.01*(avg fuel cost)

def f(x):
    global mat, b
    newx = np.zeros((x.shape[0], 2))
    newx[:,0] = np.copy(x[:,0])
    newx[:,1] = np.mean(x[:,1:], axis = 1)
    return np.matmul(newx, mat) + b

def loss_object(n,j):
    def my_loss(y_input, yz_pred):
        y_true = tf.expand_dims(y_input[:,j],1) #shape is (batchSize,1), calculated Y_hat[n+1]
        f_j = y_input[:,-1] #shape is (batchSize,d)

        y_pred = tf.expand_dims(yz_pred[:,0],1) #shape is (batchSize,1)
        z = yz_pred[:,1:-1] #shape is (batchSize,d)

        dW = tf.random.normal(tf.shape(z)) * tf.math.sqrt(dt) # noise term

        pt1 = dt * tf.expand_dims(f_j,1) #shape is same as y_pred

        pt2 = tf.math.reduce_sum(tf.multiply(z, dW), axis = 1, keepdims = True)

        y_new = y_pred - pt1 + pt2  #shape is (batchSize,1), predicted U[n+1]

        diff = y_true - y_new #shape is (batchSize, 1)
        squared = tf.math.square(diff)
        loss = tf.reduce_mean(squared) #float, avg of (E[(y_true^j - y_new^j)^2]), for given j
        #print(F"loss shape: {loss.shape}")
        return loss

    return my_loss

def train(n, j, flag, filepath):
    n_epochs = 151 #epoch = one pass through training data

    #layer sizes
    n_inputs = d
    n_hidden1 = d + 10
    n_hidden2 = d + 10
    n_hidden3 = d + 10
    n_hidden4 = d + 10
    n_outputs = 1 + d # used to be J, but had difficulties training with the loss function, d is the shape of D_X(Y)

    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(n_hidden1, input_shape=(d,), name="hidden1"+str(n), activation='tanh'),
    tf.keras.layers.Dense(n_hidden2, name="hidden2"+str(n), activation='tanh'),
    tf.keras.layers.Dense(n_hidden3, name="hidden3"+str(n), activation='tanh'),
    #tf.keras.layers.Dense(n_hidden4, name="hidden4"+str(n), activation='tanh'),
    tf.keras.layers.Dense(n_outputs, name="output"+str(n))
    ])

    if flag:
        model.compile(
                optimizer='Adam',
                loss = loss_object(n,j))
        # set weights to those from previous time slice
        if n < N-1:
            model.load_weights(filepath + str(n+1) + '-' + str(j))

        if n >= (N-2):
            model.fit(
                x = X_train[n],
                y = Y_hat[n+1], #combo of previously learned Y and the relevant X points
                batch_size = batchSize,
                epochs = n_epochs*2-1,
                callbacks = [LossPrintingCallback(), LearningRateReducerCb()],
                validation_split = 0.1,
                verbose = 0)
        else:
            model.fit(
                x = X_train[n],
                y = Y_hat[n+1],
                batch_size = batchSize,
                epochs = n_epochs,
                callbacks = [LossPrintingCallback(), LearningRateReducerCb(), earlyStop],
                validation_split = 0.1,
                verbose = 0)

    else:
        model.load_weights(filepath + str(n) + '-' + str(j)).expect_partial()

    return model

def jump_loss_object(n,j):
    def my_loss(y_input, yz_pred):
        y_true = tf.expand_dims(y_input[:,j], 1) #shape is (batchSize,1), calculated Y_hat[n+1]
        f_j = y_input[:,-1] #shape is (batchSize,1)

        y_pred = tf.expand_dims(yz_pred[:,0], 1) #shape is (batchSize,1)
        z = yz_pred[:,1:-1] #shape is (batchSize,d)
        u = tf.expand_dims(yz_pred[:,-1], 1) # shape is (batchSize,1)

        pt1 = dt * tf.expand_dims(f_j,1) #shape is same as y_pred

        dW = tf.random.normal(tf.shape(z)) * tf.math.sqrt(dt) # noise term
        pt2 = tf.math.reduce_sum(tf.multiply(z, dW), axis = 1, keepdims = True)

        dN = tf.random.poisson(tf.shape(u), lambda_poisson * dt)
        pt3 = tf.multiply(u, dN)

        y_new = y_pred - pt1 + pt2 + pt3 #shape is (batchSize,1), current predicted Y[n+1]

        diff = y_true - y_new #shape is (batchSize, 1)
        squared = tf.math.square(diff)
        loss = tf.reduce_mean(squared) #float, avg of (E[(y_true^j - y_new^j)^2]), for given j

        return loss

    return my_loss

# trains with an additional term U, which is compensating for the jump behavior
def trainJumps(n, j, flag, filepath):

    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(n_hidden1, input_shape=(d,), name="hidden1"+str(n), activation='tanh'),
    tf.keras.layers.Dense(n_hidden2, name="hidden2"+str(n), activation='tanh'),
    tf.keras.layers.Dense(n_hidden3, name="hidden3"+str(n), activation='tanh'),
    tf.keras.layers.Dense(n_hidden4, name="hidden4"+str(n), activation='tanh'),
    tf.keras.layers.Dense(n_outputs, name="output"+str(n))
    ])

    if flag: # flag == true means we want to train
        model.compile(
                optimizer='Adam',
                loss = jump_loss_object(n,j))

        # set weights to those from previous time slice
        if n < N-1:
            model.load_weights(filepath + str(n+1) + '-' + str(j))

        if n >= N - 3:
            curr_epochs = n_epochs*3 - 2
        else:
            curr_epochs = n_epochs

        model.fit(
            x = X_train[n],
            y = Y_hat[n+1],
            batch_size = batchSize,
            epochs = curr_epochs,
            callbacks = callback_list,
            validation_split = 0.1,
            verbose = 0)

    else: # flag == false means we're loading the most recent previously learned model
        model.load_weights(filepath + str(n) + '-' + str(j)).expect_partial()

    return model

''' ------------------------- START OF ALGORITHM ----------------------------- '''
''' ------------------------- ****************** ----------------------------- '''

"""
# to animate from pre-existing images
file = "Animation/______"

if animateFlag:
    print("Animating....")
    visualsML.animate(anim_flag, N)
"""

file = open(fig_path + "_parameters.txt", "w")
file.write(f"""example = {example} \n
T = {T} \n
N = {N} \n
samples = {M_training} \n
myLearnRate = {myLearnRate} \n
n_epochs = {n_epochs} \n
layerSize = {layerSize} \n
n_layers = {n_layers} \n
preInitFlag = {preInitFlag} \n
jumpFlag = {jumpFlag} \n
trainFlag = {trainFlag} \n
animateFlag =  {animateFlag} \n
figFlag = {figFlag} \n
early stopping = {earlyStopFlag} \n \n""")
file.write("""Using 0.01G_t as switching cost \n
Using original jump loss aka y_guess = y_pred - pt1 + pt2 + jump
\n""")
file.close()

for trial in range(n_trials):
    print(f"Trial {trial + 1}")

    ''' --------------- TRAINING DATA GENERATION ---------------- '''
    for mode in range(J):
        switches[N, :, mode] *= mode
    if trainFlag:
        print("Generating training data")
        X_train = generateData(X_train) #[time slice][sample in set][dimension of portfolio]
    else:
        print("Loading training data")
        X_train = np.load('x')
        switches = np.load('switches')

    xmax = min((np.amax(X_train[:,:,0])//5 + 1)*5, 200)
    xmin = (np.amin(X_train[:,:,0])//5 - 1)*5
    ymax = np.ceil(np.amax(X_train[:,:,1])) + 1
    ymin = np.floor(np.amin(X_train[:,:,1])) - 1
    #bounds = [xmin, xmax, ymin, ymax]
    bounds = [5, 200, 2, 17]

    if figFlag:
        if example == 2:
            X_vis = np.zeros((M_training, 2))
            X_vis[:,0] = X_train[N, :, 0] # electricity price
            X_vis[:,1] = np.mean(X_train[N, :, 1:], axis = 1) # average fuel price (they're all acting relatively similarly I guess)
            visualsML.visualize2D(X_vis, switches[N], bounds, N, dt, fig_path)
        if example == 1:
            visualsML.visualize2D(X_train[N], switches[N], bounds, N, dt, fig_path)

    if animateFlag:
        visualsML.visualize2D(X_train[N], switches[N], bounds, N, dt, anim_path)

    ''' --------------- TIME SLICES n = N-1, ..., 1 ---------------- '''
    for n in range(N-1, -1, -1):
        print(f"-----------------n = {n}-----------------")
        start = time.time()
        # train neural network for each mode individually
        for k in range(J):
            Y_hat[n+1, :, -1] = f(X_train[n])[:,k] #shape = (M_training, )
            print(f'Mode {k}:')
            if jumpFlag:
                model = trainJumps(n, k, trainFlag,weight_path)
                #Y_jumps = trainJumps
            else:
                model = train(n, k, trainFlag,weight_path)
            if trainFlag:
                model.save_weights(weight_path + str(n)+ '-' + str(k))

            X_curr = tf.convert_to_tensor(X_train[n])
            mode_curr = model(X_curr, training = False) #shape=(M_training, 1+d) bc of Z
            Y_curr[:,k] = mode_curr[:,0] #shape should be (M_training,)
            # NOTE!! don't think Z and U need to be saved since new Z,U aren't trained w prev Z,U

            if n == 0:
                initial = tf.convert_to_tensor(x0)
                result = model(initial, training = False) #get the value function for a specific initial vector
                print(result[0,0])
                final[k] = result[0,0]

        #determine whether current regime is optimal by comparing to all other regimes
        curr = np.empty_like(Y_curr)
        for i in range(J): #iterating over all possible "starting modes"
            curr[:,i] = Y_curr[:,i]
            for j in range(J): #iterate over other modes that could be switched too
                if j != i: # don't need to test original mode since already set
                    curr[:,j] = Y_curr[:, j] - c(i,j,n) #profit in mode j, minus switching costs

            switches[n,:,i] = np.argmax(curr, axis = 1) # which mode has the highest reward for each path
            Y_hat[n, :, i] = curr[np.arange(M_training, dtype = int), switches[n,:,i]]

        print(f'Pre-switches: {np.mean(Y_curr, axis = 0)}')
        print(f'Post-switches: {np.mean(Y_hat[n, :, :J], axis = 0)}')
        timing[n] = time.time() - start
        print(f'Time: {timing[n]}')

        if animateFlag:
            visualsML.visualize2D(X_train[n], switches[n], bounds, n, dt, anim_path)

        if figFlag and (n > N-20 or n%10 ==0):#(n in [80, 180, 280, 359]):
            if example == 2:
                X_vis = np.zeros((M_training, 2))
                X_vis[:,0] = X_train[n, :, 0] # electricity price
                X_vis[:,1] = np.mean(X_train[n, :, 1:], axis = 1) # average fuel price (they're all acting relatively similarly I guess)
                visualsML.visualize2D(X_vis, switches[n], bounds, n, dt, fig_path)
            if example == 1:
                visualsML.visualize2D(X_train[n], switches[n], bounds, n, dt, fig_path)

    #end of for-loop for N, time slices

#    if trainFlag:
#        np.save('x', X_train)
#        np.save('switches', switches)
#        np.save('value_function', Y_hat)

    if animateFlag:
        print("Animating....")
        visualsML.animate(anim_path, N)
    ''' --------------- CALCULATE EXPECTED VALUE AS TRIAL OUTPUT ---------------- '''

    #final_avg = np.mean(final, axis = 0)
    M_test = M_training//2
    optimal_paths = np.zeros((N+1, M_test)) # first half of paths start at the same point, second half follow a distribution
    modes = np.ones(M_test, dtype = int) # current mode of each path, start in mode 1
    new_modes = np.ones(M_test, dtype = int)
    for n in range(N):
        x = X_train[n, :M_training//2] # first half of training data
        new_modes = switches[n,np.arange(M_test, dtype = int), modes] # update current mode
        switching_cost = (1 - np.equal(modes, new_modes)) * c(0,1,n)# corresponding switching cost
        run = f(x)*dt
        running_cost = run[np.arange(M_test, dtype = int), new_modes] # corresponding running cost for new mode
        optimal_paths[n+1] = optimal_paths[n] + running_cost - switching_cost # y isn't in running cost

    paths_avg = np.mean(optimal_paths[N])
    print(f"Pathwise Results: {paths_avg}")
    print(f"Neural Network Results: {final}")
    Y0_list[trial] = paths_avg
#end of for-loop for n_trials

value_fn = np.mean(Y0_list[:,0]) # take average over all trials to find u(0, x_0)
description = f"Power Plant Switching Problem (2D)\
Batches: {n_batches}, Batch size: {batchSize}, Normalization: {normFlag}, \
Learned Solution: {round(value_fn,6)}, Total time: {np.sum(timing)}"
print(description)
