######################################################
# Created by April Nellis, advised by Dr. Erhan Bayraktar and Dr. Asaf Cohen
# Original model: Aid, Campi, Langrene, Pham (2014)
# Model altered to a 9D model with exogenous electricity prices
# No implementation delay
# Trains each mode individually
# Learns Y, Z, and Delta Y (jumps in value function)
# Applies memory reduction to avoid storing N versions of the state variable X
######################################################
import numpy as np
import tensorflow as tf
import scipy
from scipy.stats import norm
import time
from datetime import datetime
from tensorflow import keras
import warnings
import visualsML
import matplotlib.pyplot as plt
import math
import multiprocessing as mp
from functools import partial
from itertools import repeat

# Just disables the annoying warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

warnings.simplefilter("error", RuntimeWarning)

start = time.time()
''' --------------- MODEL PARAMETERS ---------------- '''

T = 1/4 #3 months = 1/4 year
n_tech = 3 # number of technologies
d = 9 #dimension of data (D, A^i, S^0, S^i, P), i = 1,..,n_tech
N = 90 #number of time slices, N+1 points where a decision can be made
dt = T/N #1 day between decisions
J = 4 #number of operating regimes 0, ..., J-1

x0 = np.array([[0, 0, 0, 0, 20, 60, 40, 20, 120]]) #[D0, A^1_0, A^2_0, A^3_0, S^0_0, S^1_0, S^2_0, S^3_0]
D0 = 70 # starting point for seasonality, additional OU process centered at 0
x_init = np.copy(x0)
x_init[0,0] = D0
x_init[0,1:3] = 1

#C1 = np.array([50, 60]) # potential capacities, aka modes
#C2 = np.array([0, 10])
#C = np.array(np.meshgrid(C1, C2)).T.reshape(-1,2) #(n_modes per cap)^{n_tech} of capacity combos
C = np.array([[50, 10, 10],[60, 0, 10],[60, 10, 0],[70, 0, 0]]) #C[i,s] is capacity for fuel s in mode i
print(C)
total_C = np.sum(C, axis = 1)
print(total_C)
lambda_poisson = 12 #average number of jumps in electricity prices per year
lambda_exp = 15 #inverse of average intensity of jumps
P_max = 3000 # maximum electricity price
maxerror = 0

''' --------------- FLAGS ---------------- '''
normFlag = False # whether to normalize data
jumpFlag = True
trainFlag = True # whether to train or to used saved weights
figFlag = True
earlyStopFlag = False
preInitFlag = False
if preInitFlag:
    X0 = np.load(f'x_init.npy')[N]
else:
    X0 = x0

''' --------------- NN STRUCTURE ---------------- '''
batchSize = 1000 #size of each minibatch
n_batches = 100 #number of batches
M_training = batchSize*n_batches # Size of the training set
myLearnRate = 1e-4
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
        elif n >= (N/2): self.model.optimizer.lr.assign(myLearnRate)
        else: self.model.optimizer.lr.assign(myLearnRate/10)

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
X_curr = np.zeros((M_training, d)) #[sample in set][dimension of portfolio], unprocessed availability
save_interval = 10
X_saved = np.zeros((N//save_interval + 1, d, M_training)) # number of saved X values, which reduces numerical error from inverse

Y_hat = np.zeros((M_training, J + 1)) # prev learned y and new running cost -> [time slice][data entry][y^0, ..., y^{J-1}| f_j ]
# last element gets changed with each mode that is trained

Y_curr = np.zeros((M_training, J)) # y at current time step and mode BEFORE comparing to other modes
switches = np.ones((N+1, M_training, J), dtype = int) # optimal mode for each X at a given time
final = np.zeros(J)

listD = np.zeros((2, N+1, M_training))
avgP = np.zeros(N+1)

''' --------------- FILE PATHS ---------------- '''

now = datetime.now()
dt_string = now.strftime("%m-%d-%H%M%S")

fig_path = f"Figures/{dt_string}" # file paths for figures to keep
weight_path = './Weights/weights2D-Z-' # file path for model weights

''' --------------- FUNCTIONS ---------------- '''
def setX(aX):
    # in this situation, aX has shape (M_training, d) instead of (N+1, M_training, d)
    rng_state_list = [None] * N

    # parameters for Z and A, OU processes
    alpha = np.array([[4], [8], [8], [8]])
    beta = np.array([[15, 0.1, 0.1, 0], [0.1, 0.5, -0.1, 0], [0.1, -0.1, 0.5, 0], [0, 0, 0, 0.5]])

    # parameters for technology prices
    s_long = np.array([[-4, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 2, -1, 0, 1],[0, 0, 0, 0, 0], [0, 1, 1, 1, -1]])
    s_alpha = np.expand_dims(np.array([0.4, 0.0, 0.8, 0.0, 0]), axis = 1)
    s_sigma = np.array([[2.5, 1.25, 1.25, 1.25, 1.25], [1.25, 5, 1.25, 1.25, 1.25],[1.25, 1.25, 15, 1.25, 1.25],[0.25, 0.25, 0.25, 1.5, 1.25], [1.25, 1.25, 1.25, 1.25, 3]])/100 * np.sqrt(dt)

    for n in range(N):
        rng_state_list[n] = np.random.get_state()
        dW = np.random.normal(size = (d, M_training))
        deltaN = np.random.binomial(1, lambda_poisson * dt, size = (2+n_tech, M_training)) #Poisson process with rate jump_lambda * dt
        jump_size = np.random.exponential(1/lambda_exp, size = (2+n_tech, M_training))

    # initialize array
    half = aX.shape[0]//2
    rest = aX.shape[0] - half

    # first half are starting at initial point of interest x0
    aX[:half] = x0

    # rest are starting at some distribution of points defined by X0 if preInitFlag = True
    # else, rest are also set at x0
    if X0.shape[0] < half:
        rep = half//X0.shape[0]
        leftover = rep*X0.shape[0]
        aX[half:(half+leftover)] = np.tile(X0, (rep, 1))
        aX[(half+leftover):] = X0[0] #set remaining values at initial time slice equal to first element of X0
    else:
        aX[half:] = X0[:rest]

    aX = np.transpose(aX)

    profit = np.zeros(M_training)
    curr_mode = 3*np.ones(M_training)

    #generate random paths
    for n in range(N):
        np.random.set_state(rng_state_list[N - n - 1])
        dW = np.random.normal(0, 1, size = (d, M_training))
        deltaN = np.random.binomial(1, lambda_poisson * dt, size = (2+n_tech, M_training)) #Poisson process with rate jump_lambda * dt
        jump_size = np.random.exponential(1/lambda_exp, size = (2+n_tech, M_training))

        # define the basic OU processes Z for demand and availability (dim = 1 + f)
        Z_drift = -alpha * aX[:(1+n_tech)] * dt
        Z_volatility = np.matmul(beta, dW[:(1+n_tech)])*np.sqrt(dt)
        #MZaux .* X(:,1:(1+d)) + sqrth * normrnd(0,1,[M 1+d]) * Zsigma;

        aX[:(1+n_tech)] = aX[:(1+n_tech)] + Z_drift + Z_volatility

        # define prices processes for technologies (dim = 1 + n_tech)
        S_drift = np.matmul(s_alpha * s_long, aX[(1+n_tech):]) * dt
        S_volatility = aX[(1+n_tech):] * np.matmul(s_sigma, dW[(1+n_tech):])

        aX[(1+n_tech):] = aX[(1+n_tech):] + S_drift + S_volatility
        jump = np.exp(jump_size)**deltaN # prices experience jumps, size is (M_training, 1+n_tech)
        aX[(1+n_tech):] = aX[(1+n_tech):] * jump

        #print(f"at {n+1}, max price is ${np.amax(aX[2+n_tech])} (fuel 1) and ${np.amax(aX[3+n_tech])} (fuel 2)")

        if (n+1)%save_interval == 0:
            X_saved[(n+1)//save_interval] = aX

    final_rng_state = rng_state_list[0]
    aX = np.transpose(aX)
    aX = np.float32(aX)

    return aX, final_rng_state

def getX(aX, seed,n):
    global maxerror

    # parameters for Z and A, OU processes
    alpha = np.array([[4], [8], [8], [8]])
    beta = np.array([[15, 0.1, 0.1, 0], [0.1, 0.5, -0.1, 0], [0.1, -0.1, 0.5, 0], [0, 0, 0, 0.5]])

    # parameters for technology prices
    s_long = np.array([[-4, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 2, -1, 0, 1],[0, 0, 0, 0, 0], [0, 1, 1, 1, -1]])
    s_alpha = np.expand_dims(np.array([0.4, 0.0, 0.8, 0.0, 0]), axis = 1)
    s_sigma = np.array([[2.5, 1.25, 1.25, 1.25, 1.25], [1.25, 5, 1.25, 1.25, 1.25],[1.25, 1.25, 15, 1.25, 1.25],[0.25, 0.25, 0.25, 1.5, 1.25], [1.25, 1.25, 1.25, 1.25, 3]])/100 * np.sqrt(dt)

    aX = np.transpose(aX) # now dimenson is d x M_training

    np.random.set_state(seed)

    dW = np.random.normal(0, 1, size = (d, M_training))
    deltaN = np.random.binomial(1, lambda_poisson * dt, size =(2+n_tech, M_training)) #Poisson process with rate jump_lambda * dt
    jump_size = np.random.exponential(1/lambda_exp, size =(2+n_tech, M_training))

    new_seed = np.random.get_state()

    # undo jumps
    jump = np.exp(jump_size)**deltaN
    aX[(1+n_tech):] = aX[(1+n_tech):]/jump

    Z_volatility = np.matmul(beta, dW[:(1+n_tech)]) * np.sqrt(dt)
    # MZaux = repmat(1-Zalpha*h,M,1);
    # X(:,1:(1+d)) = ( X(:,1:(1+d)) - sqrth * normrnd(0,1,[M 1+d]) * Zsigma ) ./ MZaux;
    aX[:(1+n_tech)] = (aX[:(1+n_tech)] - Z_volatility)/(1 - alpha*dt)

    #MSalpha = ( eye(1+d) + diag(Salpha) * Scomb * h )
    s_drift = np.identity(2 + n_tech) + s_alpha * s_long * dt

    #MSaux = MSsigma / MSalpha;
    s_aux = np.linalg.solve(s_drift, s_sigma) # more precise
    #print("s_aux solves s_drift * x = s_sigma")
    #print(s_aux)

    #s_inv = np.linalg.inv(s_drift)
    #extra2 = np.matmul(s_inv, aX[(1+n_tech):])
    #print("extra2 is inv(s_drift) * Y")
    #print(extra2)
    y_aux = np.linalg.solve(s_drift, aX[(1+n_tech):])
    #print("y_aux solves s_drift * x = Y")
    #print(y_aux)


    # X(:,(d+2):dfull) = (X(:,(d+2):dfull)/MSalpha)./ (Mones + normrnd(0,1,[M 1+d]) * MSaux );
    aX[(1+n_tech):] = y_aux/(1 + np.matmul(s_aux, dW[(1+n_tech):]))

    #print(f"n = {n}")
    #print(aX)

    if n%save_interval == 0 and n > 0: # retrieve X from saved list
        newX = X_saved[n//save_interval]
        maxerror = max(np.mean((newX - aX)/newX), maxerror)
        aX = newX

    aX = np.transpose(aX)

    return aX, new_seed

# seasonality shifts
ZseasMax = np.array([1.00, 0.87, 0.87, 0.9])
ZseasMin = np.array([0.70, 0.67, 0.67, 0.7])

ZseasMaxTrans = norm.ppf(ZseasMax)
ZseasMinTrans = norm.ppf(ZseasMin)

# put the demand back to normal
ZseasMaxTrans[0] = ZseasMax[0]
ZseasMinTrans[0] = ZseasMin[0]

ZseasSum = ZseasMaxTrans+ZseasMinTrans
ZseasDif = ZseasMaxTrans-ZseasMinTrans
Zshift = np.array([0, 0, 0, 0])

Dgrowth = 0.0 # yearly linear demand growth

#               Mon           Tue       Wed       Thu       Fri         Sat         Sun
Dweek=np.array([0.825, 0.975, 0.9, 1.0, 0.9, 1.0, 0.9, 1.0, 0.9, 0.975, 0.825, 0.85, 0.8, 0.85])

def getAdjX(aX, n):
    aX = np.copy(aX) # prevent changing the array passed in
    Zseas = 0.5*ZseasSum + 0.5*ZseasDif * np.cos( 2*math.pi*n*dt - Zshift )

    #Xfull(:,1) = X(:,1) + ( D0+Dgrowth*t ) * Zseas(1) * Dweek(w);
    aX[:,0] = aX[:,0] + (D0 + Dgrowth*n*dt)*Zseas[0]#*Dweek[n//14]

    #Xfull(:,2:(1+d)) = normcdf( X(:,2:(1+d)) + repmat(Zseas(2:(1+d)),M,1) );
    aX[:, 1:(1 + n_tech)] = aX[:, 1:(1 + n_tech)] + Zseas[1:(1 + n_tech)]
    aX[:, 1:(1+ n_tech)] = norm.cdf(aX[:, 1:(1+ n_tech)])

    return aX

def g_hat(X):
    return np.zeros_like(X)

def c(i,j,aX):
    S = aX[:, (1+n_tech+1):]
    # i is current mode (rows) and j is new mode (columns)
    coef = np.array([[[0,0,0,0],[1,1,0,0],[1,0,15,0],[1,1,15,0]],
                    [[1,1,0,0],[0,0,0,0],[0,1,15,0],[1,1,15,0]],
                    [[1,0,5,0],[0,1,5,0],[0,0,0,0],[1,1,0,0]],
                    [[1,1,5,0],[1,0,5,0],[1,1,0,0],[0,0,0,0]]])

    cost = 0.1*np.sum(coef[i,j]*S, axis = 1)

    return cost # size = (M_training,)

def f(aX):
    D = aX[:,0] # demand
    A = aX[:, 1:(1+n_tech)] # availability
    CO2 = aX[:, (1+n_tech)] # CO2 prices
    S = aX[:, (2+n_tech):-1] # technology prices
    price = aX[:, -1] # electricity price

    h_CO2 = np.array([[0.5], [2], [0]]) # CO2 production for technologies
    h_tech = np.array([1.0, 1.5, 1.5]) # heat rates for technologies, natural gas = 6,654 Btu/kWh, coal = 10,300 Btu/kWh, nuclear = 10,446 Btu/kWh

    profit = np.zeros((M_training, J))

    for j in range(J):
        S2 = np.transpose(h_CO2*CO2) + h_tech * S # shape (M_training, n_tech)

        curr_C = C[j] * A # shape (M_training, n_tech)

        curr_total_C = np.sum(curr_C, axis = 1) # shape (M_training,)

        '''
        # calculate fossil fuel element of price
        if j == 1 or j == 3: # second plant is turned on
            m1 = S2[:,1]/curr_C[:,0]
            cost1 = m1 * D + S2[:,0] # increases linearly with D just for simplicity
            bool1 = (D <= curr_C[:,0]) * (0 < D)
            price1 = np.where(bool1, cost1, 0) # only need to use cheaper tech

            m2 = S2[:,1]/curr_C[:,1]
            cost2 = np.minimum(m2*(D - curr_C[:,0]) + S2[:,1] + S2[:,0], P_max) # increases linearly with D just for simplicity
            bool2 = (D > curr_C[:,0]) * (D <= curr_C[:,0] + curr_C[:,1])
            price2 = np.where(bool2, cost2, 0) # need to use more expensive tech
        else:
            upper_bound = 5*S2[:,0]
            m1 = upper_bound/np.maximum(curr_C[:,0], 1e-2)
            cost1 = np.minimum(m1 * D, upper_bound) + S2[:,0]
            bool = (0 < D)
            price1 = np.where(bool,cost1, 0) # only need to use cheaper tech
            price2 = 0

        # calculate nuclear element of price
        m3 = S2[:,2]/curr_C[:,2]
        bool3 = (D > curr_C[:,0] + curr_C[:,1])
        cost3 = np.minimum(S2[:,0] + S2[:,1] + S2[:,2], P_max)
        price3 = np.where(bool3, cost3, 0)

        price[:,j] = price1 + price2 + price3
        '''

        fixed_cost = np.sum(C[j] * S2, axis = 1) # amount you're spending on production indep of demand, (M_training, n_tech)
        revenue = np.minimum(D, curr_total_C) * price
        #penalty = np.maximum(D - curr_total_C, 0) * P_max # charged a penalty of $P_max per unmet GW of electricity I guess

        profit[:,j] = revenue - fixed_cost #- penalty

    return profit, price #want to return two arrays of shape (M_training, J)

# alternate running cost suggested by Reviewer 1
def f2(aX):
    D = aX[:,0] # demand
    A = aX[:, 1:(1+n_tech)] # availability
    CO2 = aX[:, (1+n_tech)] # CO2 prices
    S = aX[:, (2+n_tech):-1] # technology prices
    price = aX[:, -1] # electricity price

    h_CO2 = np.array([[0.5], [2], [0]]) # CO2 production for technologies
    h_tech = np.array([1.0, 1.5, 1.5]) # heat rates for technologies, natural gas = 6,654 Btu/kWh, coal = 10,300 Btu/kWh, nuclear = 10,446 Btu/kWh

    profit = np.zeros((M_training, J))

    for j in range(J):
        S2 = np.transpose(h_CO2*CO2) + h_tech * S # shape (M_training, n_tech)

        curr_C = C[j] * A # shape (M_training, n_tech)

        curr_total_C = np.sum(curr_C, axis = 1) # shape (M_training,)

        fixed_cost = np.sum(C[j] * S2, axis = 1) # amount you're spending on production indep of demand, (M_training, n_tech)
        revenue = np.minimum(D, curr_total_C) * price
        #penalty = np.maximum(D - curr_total_C, 0) * P_max # charged a penalty of $P_max per unmet GW of electricity I guess

        underproduce = np.maximum(D - curr_total_C, 0) * price*2 # buy at premium
        overproduce = np.maximum(curr_total_C - D, 0) * price/2 # sell at discount

        profit[:,j] = revenue - fixed_cost + overproduce - underproduce

    return profit, price #want to return two arrays of shape (M_training, J)

def jump_loss_object(n,j):
    def my_loss(y_input, yz_pred):
        y_true = tf.expand_dims(y_input[:,j], 1) #shape is (batchSize,1), calculated Y_hat[n+1,j]
        #x = y_input[:,-d:] #shape is (batchSize,d)
        f_j = tf.expand_dims(y_input[:,-1], 1) #shape is (batchSize,1)

        y_pred = tf.expand_dims(yz_pred[:,0], 1) #shape is (batchSize,1)
        z = yz_pred[:,1:-1] #shape is (batchSize,d)
        u = tf.expand_dims(yz_pred[:,-1], 1) # shape is (batchSize,1)

        #print(y_true.shape, f_j.shape, y_pred.shape, z.shape, u.shape)
        #pt1 = dt * tf.expand_dims(f(n, x, j),1) #shape is same as y_pred
        pt1 = dt *  f_j #shape should be same as y_pred

        dW = tf.random.normal(tf.shape(z)) * tf.math.sqrt(dt) # noise term
        pt2 = tf.math.reduce_sum(tf.multiply(z, dW), axis = 1, keepdims = True)

        dN = tf.random.poisson(tf.shape(u), lambda_poisson * dt) # jump noise term
        pt3 = tf.multiply(u, dN) - u*lambda_poisson * dt

        #print(pt1.shape, pt2.shape, pt3.shape)
        y_new = y_pred - pt1 + pt2 + pt3 #shape is (batchSize,1), current predicted Y[n+1]

        diff = y_true - y_new #shape is (batchSize, 1)
        squared = tf.math.square(diff)
        loss = tf.reduce_mean(squared) #float, avg of (E[(y_true^j - y_new^j)^2]), for given j

        return loss*dt

    return my_loss

# trains with an additional term U, which is compensating for the jump behavior
def trainJumps(n, j, flag, filepath):
    init = tf.keras.initializers.GlorotUniform()

    model = tf.keras.models.Sequential([
    #tf.keras.layers.BatchNormalization(input_shape=(d,), axis=-1, momentum=0.99),
    tf.keras.layers.Dense(n_hidden1, input_shape=(d,), name="hidden1"+str(n), activation='tanh', kernel_initializer = init),
    tf.keras.layers.Dense(n_hidden2, name="hidden2"+str(n), activation='tanh', kernel_initializer = init),
    tf.keras.layers.Dense(n_hidden3, name="hidden3"+str(n), activation='tanh', kernel_initializer = init),
    tf.keras.layers.Dense(n_hidden4, name="hidden4"+str(n), activation='tanh', kernel_initializer = init),
    tf.keras.layers.Dense(n_outputs, name="output"+str(n), kernel_initializer = init)
    ])

    if flag: # flag == true means we want to train
        model.compile(
            optimizer='Adam',
            loss = jump_loss_object(n,j))

        # set weights to those from previous time slice
        if n < N-1:
            model.load_weights(filepath + str(n+1) + '-' + str(j))

        if n >= N - 5:
            curr_epochs = n_epochs*3 - 1
        else:
            curr_epochs = n_epochs

        model.fit(
            x = X_train,
            y = Y_hat,
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
file.write(f"""T = {T} \n
N = {N} \n
samples = {M_training} \n
myLearnRate = {myLearnRate} \n
n_epochs = {n_epochs} \n
layerSize = {layerSize} \n
n_layers = {n_layers} \n
preInitFlag = {preInitFlag} \n
jumpFlag = {jumpFlag} \n
trainFlag = {trainFlag} \n
figFlag = {figFlag} \n
early stopping = {earlyStopFlag} \n
modes = {C} \n""")
file.close()

'''x = np.array([[0,0,0,0,0,1,2,3,4]])
print(c(0,0,x))
print(c(0,1,x))
print(c(1,1,x))
print(c(0,3,x))'''


for trial in range(n_trials):
    print(f"Trial {trial + 1}")

    ''' --------------- TRAINING DATA GENERATION ---------------- '''
    for mode in range(J):
        switches[:, :, mode] *= mode
    if trainFlag:
        print("Generating training data")
        #X_train = generateData(X_train) #[time slice][sample in set][dimension of portfolio]
        X_curr, rng_seed = setX(X_curr) #unprocessed X at time N
        np.save('x_mr.npy', X_curr)
        rng_temp = np.append(rng_seed[1], [rng_seed[2], rng_seed[3], rng_seed[4]])
        np.save('rng.npy', rng_temp)
    else:
        print("Loading training data")
        X_curr = np.load('x_mr.npy')
        rng_temp = np.load('rng.npy')
        rng_seed = ('MT19937', rng_temp[:624], int(rng_temp[624]), int(rng_temp[625]), rng_temp[626]) # reconstruct rng state
        #switches = np.load('switches')

    listD[1, N] = np.around(X_curr[:,0], 4)
    X_train = getAdjX(X_curr, N) # this is what we use to actually train

    curr_profit, curr_price = f(X_train)
    print(f"""avg marginal profit is {np.round(np.mean(curr_profit, axis = 0)*dt, 3)}
    with variance {np.round(np.var(curr_profit*dt, axis = 0),4)}""")

    method = 1 # 0 for average value in figures, 1 for mode (most frequent) value in figures

    n = N
    labels1 = ['Switching Strategy', 'Electricity ($/Mwh)', 'Gas ($/MMBtu)', 'Mode 1', 'Mode 2', 'Mode 3', 'Mode 4']
    labels2 = ['Switching Strategy', 'Electricity ($/Mwh)', 'Coal ($/MT)', 'Mode 1', 'Mode 2', 'Mode 3', 'Mode 4']
    labels3 = ['Switching Strategy', 'Electricity ($/Mwh)', 'Nuclear ($/kg)', 'Mode 1', 'Mode 2', 'Mode 3', 'Mode 4']

    visualsML.gridHeatmap(X_train[:,-1], X_train[:,2+n_tech], switches[n] + 1, labels1, method, n, fig_path)
    visualsML.gridHeatmap(X_train[:,-1], X_train[:,3+n_tech], switches[n] + 1, labels2, method, n, fig_path)
    visualsML.gridHeatmap(X_train[:,-1], X_train[:,4+n_tech], switches[n] + 1, labels3, method, n, fig_path)


    labels11 = ['Switching Strategy', 'Coal ($/MT)', 'Gas ($/MMBtu)', 'Mode 1', 'Mode 2', 'Mode 3', 'Mode 4']
    labels21 = ['Switching Strategy', 'Nuclear ($/kg)', 'Coal ($/MT)', 'Mode 1', 'Mode 2', 'Mode 3', 'Mode 4']
    labels31 = ['Switching Strategy', 'Gas ($/MMBtu)', 'Nuclear ($/kg)', 'Mode 1', 'Mode 2', 'Mode 3', 'Mode 4']

    visualsML.gridHeatmap(X_train[:,3+n_tech], X_train[:,2+n_tech], switches[n] + 1, labels11, method, n, fig_path)
    visualsML.gridHeatmap(X_train[:,4+n_tech], X_train[:,3+n_tech], switches[n] + 1, labels21, method, n, fig_path)
    visualsML.gridHeatmap(X_train[:,2+n_tech], X_train[:,4+n_tech], switches[n] + 1, labels31, method, n, fig_path)



    ''' --------------- TIME SLICES n = N-1, ..., 1 ---------------- '''
    for n in range(N-1, -1, -1):
        print(f"-----------------n = {n}-----------------")
        # replace X at time n+1 with X at time n
        X_curr, rng_seed = getX(X_curr, rng_seed, n)
        #print(np.around(X_curr[:, 0], 4))
        #listD[1, n] = np.around(X_curr[:,0], 4)

        X_train = getAdjX(X_curr, n)

        #print(f"at {n}, max price is ${np.amax(X_train[:, 2+n_tech])} (fuel 1) and ${np.amax(X_train[:, 3+n_tech])} (fuel 2)")
        #print(f"number of zeros: {np.count_nonzero(X_train[:,1:(1+n_tech)] <= 1e-4)}")

        curr_profit, curr_price = f2(X_train)

        print(f"avg marginal profit is {np.round(np.mean(curr_profit, axis = 0)*dt, 3)} with variance {np.round(np.var(curr_profit*dt, axis = 0),4)}")
        #print(f"avg price is {np.round(np.mean(curr_price, axis = 0), 3)} with variance {np.round(np.var(curr_price, axis = 0),3)}")

        # currently Y_hat is set as Y_hat for time n+1, we want to train for time n
        #Y_hat[:, -d:] = X_train # X at n, used in training for Y_hat at n

        # train neural network for each mode individually
        x_gas = np.copy(X_train)
        x_gas[:,3+n_tech] = np.mean(X_train[:, 3 + n_tech])
        x_gas[:,4+n_tech] = np.mean(X_train[:, 4 + n_tech])
        gas = np.empty_like(Y_curr)

        x_coal = np.copy(X_train)
        x_coal[:,2+n_tech] = np.mean(X_train[:, 2 + n_tech])
        x_coal[:,4+n_tech] = np.mean(X_train[:, 4 + n_tech])
        coal = np.empty_like(Y_curr)

        x_nucl = np.copy(X_train)
        x_nucl[:,2+n_tech] = np.mean(X_train[:, 2 + n_tech])
        x_nucl[:,3+n_tech] = np.mean(X_train[:, 3 + n_tech])
        nuclear = np.empty_like(Y_curr)

        for k in range(J):
            print(f'Mode {k}:')
            Y_hat[:, -1] = curr_profit[:,k]

            mymodel = trainJumps(n, k, trainFlag, weight_path)

            if trainFlag:
                mymodel.save_weights(weight_path + str(n)+ '-' + str(k))

            Y_curr[:,k] = mymodel(X_train, training = False)[:,0] #shape=(M_training,)
            gas[:,k] = mymodel(x_gas, training = False)[:,0]
            coal[:,k] = mymodel(x_coal, training = False)[:,0]
            nuclear[:,k] = mymodel(x_nucl, training = False)[:,0]

            print(f"Variance of Y_curr: {np.var(Y_curr[:,k])}")
            # NOTE: I don't think Z and U need to be saved since new Z,U aren't trained w prev Z,U

            if n == 0:
                initial = tf.convert_to_tensor(x_init)
                result = mymodel(initial, training = False) #get the value function for a specific initial vector
                print(result[0,0])
                final[k] = result[0,0]

        #determine whether current regime is optimal by comparing to all other regimes
        curr = np.empty_like(Y_curr)

        focus_switch = np.zeros((n_tech, M_training, J)) # for holding some stuff constant
        fuel_curr = np.zeros((n_tech, M_training, J))

        for i in range(J): #iterating over all possible "starting modes"
            for j in range(J): #iterate over other modes that could be switched too
                curr[:,j] = Y_curr[:, j] - c(i,j,X_train) #profit in mode j, minus switching costs
                fuel_curr[0,:,j] = gas[:, j] - c(i,j,x_gas)
                fuel_curr[1,:,j] = coal[:, j] - c(i,j,x_coal)
                fuel_curr[2,:,j] = nuclear[:, j] - c(i,j,x_nucl)

            switches[n,:,i] = np.argmax(curr, axis = 1) # which mode has the highest reward for each path
            focus_switch[0,:,i] = np.argmax(fuel_curr[0], axis = 1) # which mode has the highest reward for each path
            focus_switch[1,:,i] = np.argmax(fuel_curr[1], axis = 1) # which mode has the highest reward for each path
            focus_switch[2,:,i] = np.argmax(fuel_curr[2], axis = 1) # which mode has the highest reward for each path

            Y_hat[:, i] = curr[np.arange(M_training, dtype = int), switches[n,:,i]]

        if figFlag and (n > N-20 or n%10 ==0):
            visualsML.gridHeatmap(X_train[:,-1], X_train[:,2+n_tech], focus_switch[0] + 1, labels1, method, n, fig_path)
            visualsML.gridHeatmap(X_train[:,-1], X_train[:,3+n_tech], focus_switch[0] + 1, labels2, method, n, fig_path)
            visualsML.gridHeatmap(X_train[:,-1], X_train[:,4+n_tech], focus_switch[0] + 1, labels3, method, n, fig_path)

            visualsML.gridHeatmap(X_train[:,3+n_tech], X_train[:,2+n_tech], focus_switch[0] + 1, labels11, method, n, fig_path)
            visualsML.gridHeatmap(X_train[:,4+n_tech], X_train[:,3+n_tech], focus_switch[0] + 1, labels21, method, n, fig_path)
            visualsML.gridHeatmap(X_train[:,2+n_tech], X_train[:,4+n_tech], focus_switch[0] + 1, labels31, method, n, fig_path)

        #avgP[n] = np.mean(curr_price[np.arange(M_training, dtype = int), switches[n,:,0]])

        print(f'Pre-switches: {np.mean(Y_curr, axis = 0)}')
        print(f'Post-switches: {np.mean(Y_hat[:, :J], axis = 0)}')
        print(f'Avg. gas: {np.mean(X_train[:,2+n_tech])}, avg. coal: {np.mean(X_train[:,3+n_tech])}, avg. nuclear: {np.mean(X_train[:,4+n_tech])}')

    #end of for-loop for N, time slices
    print(f"Max error using intermittent saves: {maxerror}")

    ''' --------------- CALCULATE EXPECTED VALUE AS TRIAL OUTPUT ---------------- '''
    print(f"Neural Network Results: {final}")
    Y0_list[trial] = final
#end of for-loop for n_trials

value_fn = np.mean(Y0_list[:,0]) # take average over all trials to find u(0, x_0)
end = time.time()
print(f"Time elapsed: {end - start}")
