import numpy as np
import math

inp = range(33)
layers = [len(inp), 10, 12, 13, 9]
bias = []# see what can be done
t = [0 for x in range(layers[-1])]          #true values
epoch = 1000
learn_rate = 0.1
weights = []

def f(x, str):
    if (str == "sigmoid"):
        return 1/1+math.exp(-x)
    else:
        pass

def f_(x, str):
    if (str == "sigmoid"):
        return 1/1+math.exp(-x)(1-1/1+math.exp(-x))
    else:
        pass

for x in range(len(layers)-1):
    a = np.random.rand(layers[x], layers[x+1])
    weights.append(a)
    print(np.shape(a))



activated = [np.array([range(x)]) for x in layers]
delta = [np.array([range(x)]) for x in layers[1:]]
activated[0] = inp
str = "sigmoid"

print delta.__len__()
def updateDeltas(delta):                            # acc to mse
    a = len(delta) - 1
    for x in range(a):
        for y in delta[a-x]:
            if (x == 0):
                delta[a-x] = (t[y] - a[y])*()          #base case #add softmax
            else:
                weiArr = weights[a-x][y]
                delta[a-x][y] = (np.dot(delta[a-x+1], weiArr))*f_(activated[y], str)
    return delta


def backProp(weight, eta, nodeBack, layerForw, ind):
    # nodeBack = activated[ind][y], nodeForw = delta[ind+1][x]
    # ind is index of prior layer, so input included and output excluded
    for x in range(len(weight)):
        for y in range(len(weight[x])):
            weight[x][y] -= eta*activated[ind][y]*delta[ind+1][x]
    return weight

def forwProp(activated, ind):
    # layer wise activation, ind is index of layer to be activated, input excluded
    activate = activated[ind]
    if (ind == len(activated)-1):
        for x in range(len(activate)):
            activate[x] = math.exp(np.dot(activated[ind-1], weights[ind-1][:,x]))
        sum = np.sum(activate)
        return activate/sum
    else:
        for x in range(len(activate)):
            activate[x] = f(np.dot(activated[ind-1], weights[ind-1][:,x]))
        return activate

def cost(str, activated, t):
    ret = 0
    for x in range(len(t)):
        if str == "mse":
            ret += (t[x] - activated[-1][x])**2
    return ret

for ep in range(epoch):
    for x in inp:
        activated[0] = x
        for ind_f in range(len(layers)-1):
            # for active in layers[ind_f]:
            activated[ind_f+1] = forwProp(activated, ind_f+1)
        delta = updateDeltas(delta)
        for ind_b in range(len(layers)-1):
            weights[ind_b] = backProp(weights[ind_b], learn_rate, 0, 0, activated[ind_b])

        print cost("mse", activated, t)