######################################################
# Carmona, Ludkovski 2008 Example 1 (2D) -> Implementing algorithm from paper
# No implementation delay
# 1/29/2021 - Implementing the algorithm from the paper
######################################################
import numpy as np
import scipy
import time
from datetime import datetime
import warnings
import visualsML
import math
import matplotlib.pyplot as plt

# for basis regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from numpy.polynomial.laguerre import lagfit, lagval

start = time.time()
'''------------------PARAMETERS----------------------'''
T = 1/4 # time horizon (in years) = 90 days
J = 3 # number of operating modes
dim = 2 # number of price processes, [price of electricity, price of gas]

M = 180 # number of time steps
N_p = 40000 # number of sample paths
dt = T/M
delta = 0*dt # time delay for implementation (in years) = 12 hours
D = 0 # number of time steps required between switches

basis = 'poly' # can choose 'poly' for standard polynomials or 'lag' for Laguerre polynomials (not ready yet)
split = N_p//2
rest = N_p - split

'''------------------DATA STRUCTURES------------------'''
paths = np.zeros((M, N_p, dim)) # contains all sample paths

curr_H = np.zeros((N_p,J)) # pathwise gain H for all paths

curr_EH = np.zeros((N_p,J)) # conditional expectation of H for all paths

if delta > 0:
    prev_EH = np.zeros((D, N_p, J)) # conditional expectation of H after delay (D steps)
else:
    prev_EH = np.zeros((N_p, J))

switch = np.zeros((M, N_p, J), dtype = int) # stores the optimal mode at [time][path][current mode]
# NOTE: switch[m,k,n,i]==i if no switch, therefore this replaces tau(t, x) from paper
for i in range(J):
    switch[:,:,i] = i # initialize switch array to the no-switch mode

''' --------------- FILE PATHS ---------------- '''

now = datetime.now()
dt_string = now.strftime("%m-%d-%H%M%S")

fig_path = f"Figures/{dt_string}-OG" # file paths for figures to keep
anim_path = f"Animation/{dt_string}-OG" # file path for images used in animation

animateFlag =  False # whether to generate an animation from visualizations
figFlag = True

'''------------------MODEL-SPECIFIC FUNCTIONS------------------------'''

# x = np.zeros((M, N_p, d)) is the empty array that will hold sample paths
def generatePaths2D(x):
    init = np.load(f'x_init.npy')[M]

    x[0,:split] = np.array([50.0, 6.0]) # first portion are starting at initial point of interest

    if init.shape[0] > rest:
        x[0,split:] = init[:rest]
    else:
        rep = rest//init.shape[0]
        leftover = rep*init.shape[0]
        x[0,split:(split+leftover)] = np.tile(init, (rep, 1))
        x[0, (split+leftover):] = init[0]


    lambda_poisson = 8
    lambda_exp = 10
    d = 2 # dimension of data

    #x[0,:] = x0 #initialize
    for m in range(M - 1):
        dW1 = np.random.normal(0,1,size=N_p) * np.sqrt(dt)
        dW2 = np.random.normal(0,1,size=N_p) * np.sqrt(dt)

        n_jumps = np.random.poisson(lambda_poisson * dt, size = N_p) #how many jumps from m*dt to (m+1)*dt

        total_jump = np.zeros(N_p)

        # finite difference approximation
        deltaN = np.random.binomial(1, lambda_poisson * dt, size = N_p) #Poisson process with rate jump_lambda * dt
        jump_size = np.random.exponential(1/lambda_exp, size = N_p)
        total_jump = np.exp(jump_size)**deltaN

        stepP = 5*(np.log(50) - np.log(x[m,:,0]))*dt + 0.5 * dW1
        x[m+1,:, 0] = x[m, :, 0] * (1 + stepP) * total_jump #from Glasserman, S(tau_j) = S(tau_j-)*Y_j

        stepG = 2*(np.log(6) - np.log(x[m,:,1])) * dt + 0.4*(0.8*dW1 + 0.6 * dW2)
        x[m+1, :, 1] = x[m,:,1] *  (1 + stepG)

        '''
        # exact approximation
        total_jump = np.ones(N_p) # np.zeros(N_p)

        for i in range(N_p):
            new_jumps = np.random.exponential(1/lambda_exp, size = n_jumps[i]) #xi_i
            total_jump[i] = total_jump[i] * np.prod(new_jumps)

        stepP = 5*(np.log(50) - np.log(x[m,:,0]))*dt - (0.5**3)*dt + 0.5 * dW1
        x[m+1,:, 0] = x[m, :, 0] * np.exp(stepP) * total_jump

        stepG = 2*(np.log(6) - np.log(x[m,:,1])) * dt - 0.5*(0.32**2 + 0.24**2)*dt + 0.4*(0.8*dW1 + 0.6 * dW2)
        x[m+1, :, 1] = x[m,:,1] * np.exp(stepG)

        '''
    x = np.float32(x)

    return x

# Switching cost
# i is current mode, j is mode we're switching to, m is current time step
def cost(i,j,m):
    G_t = paths[m, :, 1] #G_t is second element of path (price of natural gas at time t = m*dt)
    return 0.01 * G_t

# Running reward accumulated by operation of power plant for every mode
# Returns J-dim array for each path
# m = current time step
def phi(m):
    x = paths[m] # x = [electricity, gas] at time t = m*dt, size = (N_p, dim)

    mat = np.array([[0, 0.438, 0.876],[0, -0.438*7.5, -0.876*10]])
    b = np.array([-1, -1.1, -1.2])
    reward_array = np.matmul(x, mat) + b

    return reward_array #want to return array of shape (N_p, J)


'''-----------------ALGORITHM-SPECIFIC FUNCTIONS------------------'''
# Calculates conditional expectation of H using regression on polynomial basis functions
# m is time step
# H is the continuation values used in regression (shape = (N_p, K, J) I think??)
# whichReg determines the basis functions chosen
def regressContVal(m, H, whichReg):
    EH = np.zeros_like(H) # [paths][mode] I assume you get a conditional expectation for each sample path and mode

    if whichReg == 'poly': # create a vector with polynomial powers of P, G, and cross terms between P and G
        poly_array = PolynomialFeatures(degree = 6).fit_transform(paths[m])

        # implement regression
        for i in range(J):
            # perform regression where y = H(x^n, i) and poly_array = B(x^n) to find coefficients alpha
            poly_model = LinearRegression().fit(poly_array, H[:, i])

            # use the value of x^n_m for the nth path to find conditional expectation for that path
            EH[:, i] = poly_model.predict(poly_array)

        error = np.mean(np.abs(EH - H)) # check accuracy of prediction
        print(error)

        EH += phi(m)*dt # add continuation from DPP

        if m > M: # basically I turned this off, previously > m%10 == 0:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(x[:,0], x[:,1], H[:,0], marker='o', color = 'black')
            ax.scatter(x[:,0], x[:,1], EH[:,0], marker='^', color = 'red')
            plt.show()
            #plt.savefig(f"Figures/{dt_string}-regression-{m}.png", dpi=300)
            #plt.close()

    elif whichReg == 'lag': # alternately, use Laguerre polynomials (not ready)
        alpha = lagfit(x, H[:, i]) # I think lagfit only takes 1D x and this is 2D??
        EH[:,i] = lagval(x, alpha)
    else:
        print("What do you want from me????")

    return EH

# Calculate reward if switch is made
# m is current time step
# i is current mode
# EH is the expected continuation value, taking into account implementation delay
def switchVal(m, i, EH_post_switch):
    all_modes = np.zeros((N_p, J)) # stores regression switching cost for all samples paths and modes at m
    #delay = phiIntegral(m, m+D) # once you switch from i to j, stuck in mode j for D length of time
    for j in range(J):
        if j != i:
            all_modes[:, j] = EH_post_switch[:, j] - cost(i,j,m) #+ delay[:,j]

    return all_modes # note that all_mode[:, i] will still be all zeros, not assigned

# Updates reward for timestep m, given values at time m + 1
# H = H(m+1) = [H(x^0_m+1, 0),..,H(x^0_m+1, J);....; H(x^N_p_m+1, i)...] for all modes i, shape is (N_p, J)
# m + 1 is timestep with known values, used to calculate m
# tau contains optimal mode, given current mode, shape = (N_p, J)
def updateGains(H, tau, m):
    newH = np.empty_like(H) #[path][mode] pathwise cashflow at m, calculated from cashflow at m+1 (H)
    running_cost = phi(m)*dt

    for i in range(J):
        for n in range(N_p): # this is quite inefficient LOL
            if (tau[n, i] == i):
                newH[n,i] = H[n,i] + running_cost[n,i] # no switch, just continue
            else: # a switch to j at time m, given H at m+1
                j = tau[n, i]
                #newH[n,i] = newH[n,j] - cost(i, j, m)[n] # recursive...ruh-roh!
                newH[n,i] = H[n, j] + running_cost[n,j] - cost(i, j, m)[n] # switch and continue
    return newH

# Calculate left Riemann sum estimates
# 'lower' and 'upper' are /indices/ for lower and upper time bounds on integral
def phiIntegral(lower, upper):
    sum = dt * phi(lower)
    if lower>=upper:
        return np.zeros_like(sum)
    else:
        sum = dt * phi(lower)
        for dx in range(1, D):
            sum += dt * phi(lower + dx)
        return sum # shape is (N_p, J)

'''------------------MAIN------------------------------'''

#if dim == 2:
print("Generating data...")
paths = generatePaths2D(paths) # dimension is (N_p, M, dim)

xmax = (np.amax(paths[:,:,0])//5 + 1)*5
xmin = (np.amin(paths[:,:,0])//5 - 1)*5
ymax = np.ceil(np.amax(paths[:,:,1])) + 1
ymin = np.floor(np.amin(paths[:,:,1])) - 1
#bounds = [xmin,xmax, ymin,ymax]
bounds = [15, 200, 1, 17]

# initialize
curr_H = curr_H * 0 # pretty much unnecessary but want to demonstrate purposely initialized at 0

print("Beginning algorithm...")
# move backwards in time
for m in range(M-1, 0, -1): # to go from M-1 to 1, m+1 is known, calculating m
    print(f"----------------m = {m}-----------------")
    # curr_H is val at m+1, curr_EH is value at m
    curr_EH = regressContVal(m, curr_H, basis) # continuation value, shape is (K, N_p, J) just like curr_H, 6 basis functions

    if m + D < M: # enough time has passed for a switch to be allowed, based on delay
        # pick out expected values after a switch at m
        if D == 0:
            delayed_EH = curr_EH
        elif D == 1:
            delayed_EH = prev_EH
        else:
            delayed_EH = prev_EH[-1] # pick out the one we want right now (last one)

        for i in range(J): # find best strategy for each mode
            modeValues = switchVal(m, i, delayed_EH) #shape is (N_p, J), if j!=i calculates val after switch
            modeValues[:, i] = curr_EH[:, i] # insert continuation values for staying in mode i
            switch[m,:,i] = np.argmax(modeValues, axis = 1) #shape is (N_p), mode with largest val in each path
    else:
        for i in range(J):
            switch[m,:,i] = i

    curr_H = updateGains(curr_H, switch[m], m) # update pathwise realized cashflows

    if D == 1:
        prev_EH = curr_EH
    elif D > 1:
        prev_EH = np.roll(prev_EH, 1, axis = 0) # [m+1, m+2, ..., m+D] -> [m+D, m+1, .., m+D-1]
        prev_EH[0] = curr_EH # [m, m+1, ..., m+D-1]

    if animateFlag:
        visualsML.visualize2D(paths[m], switch[m], bounds, m, dt, anim_path)
    if figFlag and (m%10 == 0):
        visualsML.visualize2D(paths[m], switch[m], bounds, m, dt, fig_path)

print(f"Filename: {dt_string}")

result = np.mean(curr_H[:split] + phi(0)[:split]*dt, axis = 0) # average H^K + continuation over all paths, should be shape (J,)
print(f"The value of the power plant at t=0 is:")
for i in range(J):
    print(f"Mode {i}: {result[i]}")

print(f"Switching delay: {D} time steps")

if animateFlag:
    visualsML.visualize2D(paths[0], switch[0], bounds, 0, dt, anim_path)
    print("Animating....")
    visualsML.animate(anim_path, M-1)

'''
path_valfunc = np.zeros((N_p, J))
curr_mode = np.zeros(J, dtype = int) # track how modes switch over time
for i in range(J):
    curr_mode[i] = i # initialize current mode as starting modes 0, 1, 2

path_valfunc = phi(0)*dt # initialize at first step, don't switch? I guess this first step isn't that important
for n in range(N_p):
    for m in range(1, M):
        #if n ==1: print(curr_mode)
        running_cost = phi(m)*dt
        for i in range(J):
            j = switch[m, n, curr_mode[i]] # which mode to switch to
            if curr_mode[i] == j:
                profit = running_cost[n,curr_mode[i]] # no switching cost, just running cost
            else:
                profit = running_cost[n,j] - cost(curr_mode[i],j,m)[n]
                curr_mode[i] = j #update current mode

            path_valfunc[n, i] += profit
mean = np.mean(path_valfunc, axis = 0)
print(f"Pathwise: {mean}")
'''
end = time.time()
print(f"Time elapsed: {end - start}")
