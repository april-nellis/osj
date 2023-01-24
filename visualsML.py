##########################################
# Module:
# Functions for making pretty pictures
##########################################
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.axes3d import get_test_data
import matplotlib
import matplotlib.animation as animation
import matplotlib.image as mgimg
import seaborn as sns
import math
from scipy import stats

plt.rcParams.update({'font.size': 10})

def visualize2D(X, switches, bounds, n, dt, filename):
    # X = [M_training, d], M_training at a specific time slice -> X_norm[n]
    # switches = [M_training, J], preferrable mode, given a specific current mode/x value/time
    # bounds = [x_min, x_max, y_min, y_max]
    J = switches.shape[1]
    cmap = plt.get_cmap("RdYlGn")

    for mode in range(J):
        X[mode] = [0,0]
        switches[mode,:] = mode # just initialize for legend, so there's one dot of each color in every graph

    switch_colors = cmap(switches/J)

    fig, axs = plt.subplots(J, sharex = True, sharey = True)
    fig.suptitle(f"Optimal Switching Strategy at {int(n * dt * 360)} days")

    if bounds != 0:
        plt.ylim((bounds[2],bounds[3]))
        plt.xlim((bounds[0],bounds[1]))

    '''
    x_values = np.linspace(bounds[0], bounds[1], 100)
    lines = np.zeros((J, J-1, 100)) # lines[mode] should be a 2x100 array, holds the boundaries if switching cost is 0.05
    lines[:,0] = x_values/7.5
    lines[:,1] = x_values/10#x_values*0.08
    b = np.array([[-1.1/(0.438 * 7.5)],[-1.2/(0.874*10)]]) # lines for where profit is positive
    #b = np.array([[[0.15], [0.1]],[[0.05], [0.15]],[[0.05], [0.1]]])/0.438 # lines for where one mode is preferrable to another

    lines += b

    ** insert into for loop below:
                #if newmode < J - 1:
                #    b = coefs[mode, newmode, 1]
                #    m = coefs[mode, newmode, 0]
                #    x_reg = (y_reg - b)/m
                #    axs[mode].plot(x_reg, y_reg,label=f"y={round(m,3)}x + {round(b,3)}")

    '''

    for mode in range(J): # current mode
        # axs[mode].plot(x_values, np.transpose(lines[mode]), 'k:', linewidth = 0.5)
        for newmode in range(J):
            curr = X[switches[:,mode]==newmode] # at which points should we switch to newmode?
            color = switch_colors[switches[:,mode]==newmode] # [path, mode, color of optimal mode]
            axs[mode].scatter(curr[:,0], curr[:,1], c = color[:,mode,:], s = 1, label = f"Mode {newmode + 1}", alpha = 0.7, edgecolors = 'none')

        axs[mode].set_title(f"Current Mode: {mode + 1}")
        axs[mode].legend(title = "Optimal:", loc = "upper left", bbox_to_anchor = (1,1), markerscale = 5)

    plt.subplots_adjust(right = 0.70)
    plt.subplots_adjust(hspace = 0.30)

    for ax in axs:
        ax.label_outer()

    axs[J//2].set(ylabel='Avg. Fuel Price ($/MMBtu)')
    axs[J-1].set(xlabel='Price of Electricity ($/Mwh)')

    plt.savefig(f"{filename}-{n}.png", dpi=300)
    plt.close()

def animate(filename, N):
    fig = plt.figure()
    plt.axis('off')
    ims = []
    for n in range(N+1):
        img = mgimg.imread(f'{filename}-{n}.png')
        im = plt.imshow(img, animated=True)
        ims.append([im])
        #print(f"Frame {n} done")
        os.remove(f'{filename}-{n}.png')

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=100)
    ani.save(f"{filename}-movie.mp4")

def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # First get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
    for j, step in enumerate(step_list):
        if step in step_dict[key]:
            this_cdict[step] = new_LUT[j, i]
        elif new_LUT[j,i] != old_LUT[j, i]:
            this_cdict[step] = new_LUT[j, i]
    colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
    colorvector.sort()
    cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

def heatScatter2D(X, Y, modes, values, n, labels, filename):
    # X = [M_training], value of one state variable at a specific time slice
    # Y = [M_training], value of other state variable at a specific time slice
    # modes = [M_training, J], preferrable mode, given a specific current mode/x value/time
    # bounds = [x_min, x_max, y_min, y_max]
    J = modes.shape[1]

    cmap = plt.get_cmap("BuPu")
    colors = cmap((modes+1)/(J+1))

    fig, axs = plt.subplots(J + 1, sharex = True, sharey = True)
    fig.suptitle(f"Optimal Switching Strategy at step {n}")

    xmax = np.ceil(np.amax(X)) # high price, tech 1
    xmin = np.floor(np.amin(X))# low price, tech 1
    ymax = np.ceil(np.amax(Y)) # high price, tech 2
    ymin = np.floor(np.amin(Y)) # high price, tech 2
    bounds = [xmin, xmax, ymin, ymax]

    if bounds != 0:
        plt.ylim((bounds[2],bounds[3]))
        plt.xlim((bounds[0],bounds[1]))
    else:
        plt.xlim((0,400))
        plt.ylim((0,4000))

    for mode in range(J):
        axs[mode].scatter(X, Y, c = colors[:,mode,:], s = 2)
        axs[mode].set_title(f"Current Mode: {mode}")

    cmap = plt.get_cmap("mako")
    colors = cmap(values/np.amax(values))

    axs[J].scatter(X, Y, c = colors, s = 2)
    axs[J].set_title(labels[2])

    axs[J//2].set(ylabel = "Price of " + labels[1])
    axs[J].set(xlabel = "Price of " + labels[0])

    plt.subplots_adjust(hspace = 0.30)

    #axs[J//2].legend(title = "Optimal:", loc = "upper left", bbox_to_anchor = (1,1))
    #plt.subplots_adjust(right = 0.80)
    #plt.subplots_adjust(hspace = 0.30)
    #for ax in axs:
    #	ax.label_outer()
    #axs[J//2].set(ylabel='Price of Gas ($/MMBtu)')
    #axs[J-1].set(xlabel='Price of Electricity ($/Mwh)')

    plt.savefig(f"{filename}-{n}-valuefunc.png", dpi=300)
    plt.close()

def gridHeatmap(x, y, values, labels, method, n, filename):
    # Creates a collection of subplots which share the same (x,y) coordinates
    # but different 'values' (i.e. different 3rd dimension or different f(x,y)).
    # These values are illustrated using colors, hence the heatmap name.
    # The x-y plane is divided into a grid and each bin is assigned one color.

    # x - x-axis, size (M_training,)
    # y - y-axis, size (M_training,)
    # values - 2D array of size (M_training, n_val) where n_val = # of datasets
    # method - whether to aggregate values based on mean (0) or mode (1)

    n_val = values.shape[1]

    bins = 30 # how many squares in each direction
    maps = np.zeros((n_val, bins, bins)) # will use for heatmaps, order is [D, CO2, A1, A2, switch_mean, switch_mode]
    maps.fill(np.nan)

    x1_min = np.amin(x)
    x1_max = np.mean(x) + 4*np.std(x) #np.amax(x)
    x2_min = np.amin(y)
    x2_max = np.mean(y) + 4*np.std(y) #np.amax(y)

    a = (x1_max - x1_min)/bins # range of one bin for fuel 1
    b = (x2_max - x2_min)/bins # range of one bin for fuel 2

    x_tx = [None]*bins #np.linspace(x1_min, x1_max, bins, dtype = int)
    y_tx = [None]*bins #np.linspace(x2_min, x2_max, bins, dtype = int)
    x_tx[0] = round(x1_min)
    x_tx[-1] = round(x1_max)
    y_tx[0] = round(x2_min)
    y_tx[-1] = round(x2_max)

    if n_val in [1, 2, 3, 4]:
        I = 1
        J = n_val
    else:
        J = math.ceil(math.sqrt(n_val))
        if J*(J-1) >= n_val:
            I = J - 1
        else:
            I = J

    plt.rcParams["figure.figsize"] = (J*3,I*3+0.5) # width, height
    fig,axs = plt.subplots(I,J)
    fig.suptitle(f"{labels[0]} at step {n}")

    # build bins for (S_1, S_2) which are the two different fuels
    for i in range(bins):
        lower1 = np.where(x >= a*i + x1_min)
        upper1 = np.where(x < a*(i+1) + x1_min)
        select1 = np.intersect1d(lower1, upper1)

        for j in range(bins):
            lower2 = np.where(y >= b*j + x2_min)
            upper2 = np.where(y < b*(j+1) + x2_min)
            select2 = np.intersect1d(lower2, upper2)

            select = np.intersect1d(select1, select2)
            if select.shape[0] > 5: # don't want squares with only few point
                for k in range(n_val):
                    if method == 0:
                        maps[k,i,j] = np.mean(values[select, k]) # average value
                    elif method == 1:
                        maps[k,i,j] = stats.mode(values[select,k])[0] # most frequent value
                    else:
                        print('invalid method parameter')

    maps = np.transpose(maps, axes = (0, 2, 1)) # this must be done after ALL edits to maps

    avg = np.mean(values)
    maps2 = np.nan_to_num(maps, nan = avg) # don't want to fill NaN with 0 because that messes up min calculations

    # find min and max of squares so we get a nice range of colors (hopefully)
    low = 0 # np.amin(maps2) # np.mean(maps2) - 3*np.std(maps2)
    high = 4 #np.amax(maps2) #np.mean(maps2) + 3*np.std(maps2)

    # fill in the subplot grid
    for i in range(I):
        for j in range(J):
            idx = J*i + j # idx runs from 0 to n_val
            if I == 1:
                curr_ax = axs[j]
            else:
                curr_ax = axs[i,j]

            s = sns.heatmap(maps[idx], cmap = "BuPu", linewidths = 0.5, ax = curr_ax, vmin = low, vmax = high, \
                        cbar = False, xticklabels = x_tx, yticklabels = y_tx)

            #axs[i, j].tick_params(labelsize=20)
            curr_ax.invert_yaxis()
            curr_ax.set_aspect('equal')
            curr_ax.set_title(labels[3+idx])
            if i == I-1 and j == 0:
                curr_ax.set(xlabel=labels[1], ylabel=labels[2])

    #plt.subplots_adjust(right = 0.70)
    #plt.subplots_adjust(wspace = 0.30, left=0.1, bottom=0, right=0.9, top=1)
    plt.tight_layout()

    path = f"{filename}-{n}-grid.png"
    uniq = 1

    while os.path.exists(path):
      path = f"{filename}-{n}-grid({uniq}).png"
      uniq += 1

    plt.savefig(path, dpi=300)
    #plt.show()
    plt.close()

    return 0

def simpleBinHeatmap(x, y, values, labels, n, filename):
    # X - state variables, size = (M_training, 2)
    # values - whatever we're coloring the map based on, size = (M_training,)

    plt.rcParams["figure.figsize"] = (15,10)

    bins = 20 # how many squares in each direction
    map = np.zeros((bins, bins)) # will use for heatmaps, order is [D, CO2, A1, A2, switch_mean, switch_mode]
    map.fill(np.nan)

    x1_min = np.amin(x)
    x1_max = np.amax(x)
    x2_min = np.amin(y)
    x2_max = np.amax(y)

    bounds = [x1_min, x1_max, x2_min, x2_max]

    a = (x1_max - x1_min)/bins # range of one bin for fuel 1
    b = (x2_max - x2_min)/bins # range of one bin for fuel 2

    x_ticks_list = np.linspace(x1_min, x1_max, bins, dtype = int)
    y_ticks_list = np.linspace(x2_min, x2_max, bins, dtype = int)

    plt.title(labels[0])

    # build bins for (S_1, S_2) which are the two different fuels
    for i in range(bins):
        lower1 = np.where(x >= a*i + x1_min)
        upper1 = np.where(x < a*(i+1) + x1_min)
        select1 = np.intersect1d(lower1, upper1)

        for j in range(bins):
            lower2 = np.where(y >= b*j + x2_min)
            upper2 = np.where(y < b*(j+1) + x2_min)
            select2 = np.intersect1d(lower2, upper2)

            select = np.intersect1d(select1, select2)
            if select.shape[0] > 5: # don't want squares with only few point
                map[i,j] = np.mean(values[select])

    #map = np.transpose(map, axes = [1, 0]) # ensures that heatmaps are configured correctly

    avg = np.mean(values)
    map2 = np.nan_to_num(map, nan = avg) # don't want to fill NaN with 0 because that messes up min calculations

    # find min and max of squares so we get a nice range of colors (hopefully)
    low = np.mean(values) - 3*np.std(values) #np.amin(maps2[4:7])
    high = np.mean(values) + 3*np.std(values)#np.amax(maps2[4:7])

    s = sns.heatmap(map, cmap = "mako", linewidths = 0.5, annot=True, vmin = low, vmax = high, xticklabels = x_ticks_list, yticklabels = y_ticks_list)
    s.invert_yaxis()
    s.set(xlabel=labels[1], ylabel=labels[2])

    plt.savefig(f"{filename}-{n}-map.png", dpi=300)
    #plt.show()
    plt.close()

    return 0
