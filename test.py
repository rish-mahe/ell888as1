import numpy as np
import math
import random



inp = np.ones((35353, 784))                 #rows, columns = no of samples, dimension
layers = [len(inp), 10, 12, 13, 9]
bias = []# see what can be done

data = np.genfromtxt('/home/rishabh/Desktop/gzip/new_train.csv', delimiter=',')  # rows, columns = no of samples, dimension
# print data
# data = data
inp = data[:,1:]/255.

str = 'rbitnysuh'
lets = {}
i = 0
for x in str:
    lets[ord(x)-96] = i
    i+=1
# print lets

layers = [len(inp[0]), 750, 9]  # No of neurons in each layer. layers[0] being the no of neurons in input layer.
bias = []  # see what can be done
out = data[:,0]
# print out
t = []
for x in out:
    vec = [0 for i in range(layers[-1])]
    vec[lets[x]] = 1
    t.append(vec)

# print t
# print jjjj

t = np.array(t)   # true values------stored as (no of input * output dimension). Each row of this matrix tells the output corresponding to that input.
epoch = 1000
learn_rate = 0.000001
weights = []
drop = [0, 0.5, 0.5, 0.5, 0]  # drop out probabilities for each layer starting from input layer. No dropout in input as well as output layer.

# This is a function for the activation functions.It takes x as an input for to be used in activation function and str inducates which activation function will be used


def f(x, str):
    if (str == "sigmoid"):
        return 1 / (1 + np.exp(-x))
    else:
        pass


# This is the derivative for the activation function.
def f_(x, str):
    if (str == "sigmoid"):
        return (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))
    else:
        pass


# This creates an initial random 3-d matrix of weights and biases.
for x in range(len(layers) - 1):
    a = np.random.rand(layers[x], layers[x + 1])
    b = np.random.rand(layers[x + 1])

    weights.append(a)
    bias.append(b)
    # print(np.shape(a))

activated = [np.ones((len(inp), x), dtype=float) for x in layers]
delta = [np.array(range(x), float) for x in layers[1:]]
# print delta[3][10]
activated[0] = inp

# print delta.__len__()


def updateDeltas(delta):  # acc to mse
    # print "updating deltas"
    a = len(delta)
    for x in range(a):
        if (x == 0):
            matr = np.exp(-np.dot(activated[-2], weights[-1]))
            matr = matr/np.sum(matr, axis=1, keepdims=True)
            delta[a-x-1] = np.sum((t - activated[-1])*matr*(1-matr), axis=0)
            # print delta[a-x-1]

        else:
            for y in range(len(delta[a - x - 1])):
                weiArr = weights[a - x][y]
                # print weights[a - x]
                lh = (np.dot(delta[a - x], weiArr))
                rh = (np.sum(f_(np.dot(activated[a - x - 1], weights[a - x - 1][:,y]), "sigmoid")))
                # print delta[a - x - 1][y], "1", np.shape(lh * rh)
                delta[a - x - 1][y] = lh*rh
                # print delta[a - x - 1][y], "1", np.shape(lh*rh)

                # print lh, ":3", np.shape(lh)
                # print rh, "4", np.shape(rh)

    return delta


def thresh(a, l):
    if a < drop[l]:
        return 0
    else:
        return 1


def backProp(weight, eta, nodeBack, layerForw, ind, bias):
    # nodeBack = activated[ind][y], nodeForw = delta[ind+1][x]
    # ind is index of prior layer, so input included and output excluded
    # print "backprop"
    for x in range(len(weight)):
        for y in range(len(weight[x])):
            if(store[ind+1,x]==1):
                weight[x][y] -= eta * np.sum(activated[ind], axis=0)[x] * delta[ind][y]
                # print eta * np.sum(activated[ind], axis=0)[x] * delta[ind][y]
                # print x, y, ind
                bias -= delta[ind]
            else:
                pass
    return weight, bias


store = np.zeros(len(layers),max(layers))
def forwProp(activated, ind):
    # layer wise activation, ind is index of layer to be activated, input excluded
    activate = activated[ind]
    if (ind == len(activated) - 1):
        for row in range(len(activate)):
            for x in range(len(activate[row])):
                state = thresh(random.uniform(0, 1),ind)
                activate[row][x] = math.exp(-1 * (np.dot(activated[ind - 1][row], weights[ind - 1][:, x]) + bias[ind - 1][x]))*state
                store[row,x] = state

        return activate / np.sum(activate, axis=1, keepdims=True)
    else:
        for row in range(len(activate)):
            for x in range(len(activate[row])):
                state = thresh(random.uniform(0, 1),ind)
                activate[row][x] = f((np.dot(activated[ind - 1][row], weights[ind - 1][:, x]) + bias[ind - 1][x]), "sigmoid")*state
                store[row,x] = state
        return activate

def cost(str):
    ret = 0
    if (str == "mse"):
        ret = np.sum((t - activated[-1]) ** 2)
    return ret


costs = []

for ep in range(epoch):
    activated[0] = inp
    for ind_f in range(len(layers) - 1):
        # for active in layers[ind_f]:
        activated[ind_f + 1] = forwProp(activated, ind_f + 1)
    delta = updateDeltas(delta)
    print cost("mse")/len(data), "error, Interation: ", ep
    costs.append(cost("mse")/len(data))
    for ind_f in range(len(layers) - 1):
        # for active in layers[ind_f]:
        weights[ind_f], bias[ind_f] = backProp(weights[ind_f], learn_rate, 0, 0, ind_f, bias[ind_f])

print "One last time"
for ind_f in range(len(layers) - 1):
    # for active in layers[ind_f]:

    activated[ind_f+1] = forwProp(activated, ind_f+1)
delta = updateDeltas(delta)
