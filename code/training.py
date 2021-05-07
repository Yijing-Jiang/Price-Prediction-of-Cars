import numpy as np
import pandas as pd
import utils
import modules
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from mpl_toolkits.mplot3d import Axes3D
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score


# load training data
(X, y) = utils.load("/Users/mac-pro/Desktop/20Fall/EE660/HW/Final/code/training.csv")

'''
-----------------------------------------------------------------------
Using training dataset to find the best parameters of each modules, 
and to choose the best module by comparing between different modules.
'''
# MLE on all features[baseline], 0138 features, all features with poly (d = 2), all features with PCA (c = 7)
T = 10
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True,random_state=None)
#save the result:(coef, Etrain, Eval)
result_all = []
result_4 = []
result_all_poly = []
result_all_pca = []
for t in range(T):
    for tr_index, v_index in skf.split(X, y):
        #training data
        X_tr = X[tr_index,:]
        y_tr = y[tr_index]
        #validation data
        X_val = X[v_index,:]
        y_val = y[v_index]
        #standardize
        (X_tr_std, X_val_std) = utils.standardizing(X_tr, X_val)
        #MLE on all features:
        result1 = modules.MLE(X_tr_std, y_tr, X_val_std, y_val)
        result_all.append(result1)
        #MLE on four features:
        f = [0,1,3,8]
        result2 = modules.MLE(X_tr_std[:,f], y_tr, X_val_std[:,f], y_val)
        result_4.append(result2)
        #MLE on all features with poly 2:
        d = 2
        result3 = modules.MLE_poly(d, X_tr_std, y_tr, X_val_std, y_val)
        result_all_poly.append(result3)
        #MLE on all features with pca 7:
        c = 7
        result4 = modules.MLE_pca(c, X_tr_std, y_tr, X_val_std, y_val)
        result_all_pca.append(result4)
print("-------")
print("MLE on all features: mean var")
print("Etrain " + str(np.mean(np.array(result_all)[:,3])) + " " + str(np.var(np.array(result_all)[:,3])))
print("Eval " + str(np.mean(np.array(result_all)[:,4])) + " " + str(np.var(np.array(result_all)[:,4])))
print("-------")
print("MLE on 0138 features: mean var")
print("Etrain " + str(np.mean(np.array(result_4)[:,3])) + " " + str(np.var(np.array(result_4)[:,3])))
print("Eval " + str(np.mean(np.array(result_4)[:,4])) + " " + str(np.var(np.array(result_4)[:,4])))
print("-------")
print("MLE on all features with poly 2: mean var")
print("Etrain " + str(np.mean(np.array(result_all_poly)[:,3])) + " " + str(np.var(np.array(result_all_poly)[:,3])))
print("Eval " + str(np.mean(np.array(result_all_poly)[:,4])) + " " + str(np.var(np.array(result_all_poly)[:,4])))
print("-------")
print("MLE on all features with pca 7: mean var")
print("Etrain " + str(np.mean(np.array(result_all_pca)[:,4])) + " " + str(np.var(np.array(result_all_pca)[:,4])))
print("Eval " + str(np.mean(np.array(result_all_pca)[:,5])) + " " + str(np.var(np.array(result_all_pca)[:,5])))


# Ridge regression: choose best lambda
L = [0.1, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 100]
T = 10
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True,random_state=None)
#save the result:
result = []
for t in range(T):
    for tr_index, v_index in skf.split(X, y):
        #training data
        X_tr = X[tr_index,:]
        y_tr = y[tr_index]
        #validation data
        X_val = X[v_index,:]
        y_val = y[v_index]
        #standardize
        (X_tr_std, X_val_std) = utils.standardizing(X_tr, X_val)
        #Ridge regression on all features:
        result_L = []
        for l in L:
            result_l = modules.ridge(l, X_tr_std, y_tr, X_val_std, y_val)
            result_L.append(result_l)
        result.append(result_L)
mean_Etrain = np.mean(np.array(result)[:,:,3], axis = 0)
var_Etrain = np.var(np.array(result)[:,:,3], axis = 0)
mean_Eval = np.mean(np.array(result)[:,:,4], axis = 0)
var_Eval = np.var(np.array(result)[:,:,4], axis = 0)
best_lambda = L[np.argmin(mean_Eval)]
print("Ridge Regression best lambda: " + str(best_lambda)+" "+str(mean_Etrain[np.argmin(mean_Eval)])+" "+str(mean_Eval[np.argmin(mean_Eval)]))

plt.plot(np.log10(L), mean_Etrain, 'b', label = 'Etrain')
plt.plot(np.log10(L), mean_Eval, 'r', label = 'Eval')
plt.title("Ridge Regression best lambda: " + str(best_lambda)+" "+str(mean_Etrain[np.argmin(mean_Eval)])+" "+str(mean_Eval[np.argmin(mean_Eval)]))
plt.legend()
plt.show()


# Lasso regression (l is the lambda in course;)
L = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 80, 100]
T = 10
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True,random_state=None)
#save the result:
result = []
for t in range(T):
    for tr_index, v_index in skf.split(X, y):
        #training data
        X_tr = X[tr_index,:]
        y_tr = y[tr_index]
        #validation data
        X_val = X[v_index,:]
        y_val = y[v_index]
        #standardize
        (X_tr_std, X_val_std) = utils.standardizing(X_tr, X_val)
        #Ridge regression on all features:
        result_L = []
        for l in L:
            result_l = modules.lasso(l, X_tr_std, y_tr, X_val_std, y_val)
            result_L.append(result_l)
        result.append(result_L)
mean_Etrain = np.mean(np.array(result)[:,:,3], axis = 0)
var_Etrain = np.var(np.array(result)[:,:,3], axis = 0)
mean_Eval = np.mean(np.array(result)[:,:,4], axis = 0)
var_Eval = np.var(np.array(result)[:,:,4], axis = 0)
best_lambda = L[np.argmin(mean_Eval)]
print("Lasso Regression best lambda: " + str(best_lambda)+" "+str(mean_Etrain[np.argmin(mean_Eval)])+" "+str(mean_Eval[np.argmin(mean_Eval)]))

plt.plot(np.log10(L), mean_Etrain, 'b', label = 'Etrain')
plt.plot(np.log10(L), mean_Eval, 'r', label = 'Eval')
plt.title("Lasso Regression best lambda: " + str(best_lambda)+" "+str(mean_Etrain[np.argmin(mean_Eval)])+" "+str(mean_Eval[np.argmin(mean_Eval)]))
plt.legend()
plt.show()

# KNN (baseline)
N_K = np.arange(1, 10, 1)
weights = ['uniform', 'distance']
T = 10
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True,random_state=None)
#save the result:
result = []
for t in range(T):
    for tr_index, v_index in skf.split(X, y):
        #training data
        X_tr = X[tr_index,:]
        y_tr = y[tr_index]
        #validation data
        X_val = X[v_index,:]
        y_val = y[v_index]
        #standardize
        (X_tr_norm, X_val_norm) = utils.normalizing(X_tr, X_val)
        #Ridge regression on all features:
        result_L = []
        for w in weights:
            for n in N_K:
                result_l = modules.knn(n, w, X_tr_norm, y_tr, X_val_norm, y_val)
                result_L.append(result_l)
        result.append(result_L)

mean_Etrain = np.mean(np.array(result)[:,:,2], axis = 0)
var_Etrain = np.var(np.array(result)[:,:,2], axis = 0)
mean_Eval = np.mean(np.array(result)[:,:,3], axis = 0)
var_Eval = np.var(np.array(result)[:,:,3], axis = 0)
best_n_uniform = N_K[np.argmin(mean_Eval[0:9])]
best_n_distance = N_K[np.argmin(mean_Eval[9:18])]
print("KNN_uniform best k : " + str(best_n_uniform)+" "+str(mean_Etrain[np.argmin(mean_Eval[0:9])])+" "+str(mean_Eval[np.argmin(mean_Eval[0:9])]))
print("KNN_distance best k : " + str(best_n_distance)+" "+str(mean_Etrain[9+np.argmin(mean_Eval[9:18])])+" "+str(mean_Eval[9+np.argmin(mean_Eval[9:18])]))

plt.plot(N_K, mean_Etrain[0:9],  label = 'Etrain_uniform')
plt.plot(N_K, mean_Eval[0:9],   label = 'Eval_uniform')
plt.plot(N_K, mean_Etrain[9:18],   label = 'Etrain_distance')
plt.plot(N_K, mean_Eval[9:18],   label = 'Eval_distance')
plt.title("KNN (weights = 'uniform' or 'distance'; k = 1 to 9)")
plt.legend()
plt.show()

# CART (baseline for tree based module)
decrease = [1000, 2000, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 20000, 25000, 50000]
T = 10
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True,random_state=None)
#save the result:
result = []
for t in range(T):
    for tr_index, v_index in skf.split(X, y):
        #training data
        X_tr = X[tr_index,:]
        y_tr = y[tr_index]
        #validation data
        X_val = X[v_index,:]
        y_val = y[v_index]
        #standardize
        (X_tr_norm, X_val_norm) = utils.normalizing(X_tr, X_val)
        #cart on different min_impurity_decrerase
        result_L = []
        for d in decrease:
            result_l = modules.cart(None, d, X_tr_norm, y_tr, X_val_norm, y_val)
            result_L.append(result_l)
        result.append(result_L)
mean_Etrain = np.mean(np.array(result)[:,:,5], axis = 0)
var_Etrain = np.var(np.array(result)[:,:,5], axis = 0)
mean_Eval = np.mean(np.array(result)[:,:,6], axis = 0)
var_Eval = np.var(np.array(result)[:,:,6], axis = 0)
mean_depth = np.mean(np.array(result)[:,:,0], axis = 0)
mean_leafs = np.mean(np.array(result)[:,:,1], axis = 0)
'''
best_lambda = L[np.argmin(mean_Eval)]
print("Lasso Regression best lambda: " + str(best_lambda)+" "+str(mean_Etrain[np.argmin(mean_Eval)])+" "+str(mean_Eval[np.argmin(mean_Eval)]))
'''
plt.subplot(221)
plt.plot(np.log10(decrease), mean_Etrain, 'b', label = 'mean_Etrain')
plt.plot(np.log10(decrease), mean_Eval, 'r', label = 'mean_Eval')
plt.legend()
#plt.title("Lasso Regression best lambda: " + str(best_lambda)+" "+str(mean_Etrain[np.argmin(mean_Eval)])+" "+str(mean_Eval[np.argmin(mean_Eval)]))
plt.subplot(222)
plt.plot(np.log10(decrease), var_Etrain, 'b', label = 'var_Etrain')
plt.plot(np.log10(decrease), var_Eval, 'r', label = 'var_Eval')
plt.legend()
plt.subplot(223)
plt.plot(np.log10(decrease), mean_depth, label = 'mean_depth')
plt.legend()
plt.subplot(224)
plt.plot(np.log10(decrease), mean_leafs, label = 'mean_leafs')
plt.legend()
plt.suptitle("CART with different min_impurity_decrease")
plt.show()
'''
decrease = 10**3.5
leafs = np.arange(10,100,2)
T = 10
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True,random_state=None)
#save the result:
result = []
for t in range(T):
    for tr_index, v_index in skf.split(X, y):
        #training data
        X_tr = X[tr_index,:]
        y_tr = y[tr_index]
        #validation data
        X_val = X[v_index,:]
        y_val = y[v_index]
        #standardize
        (X_tr_norm, X_val_norm) = utils.normalizing(X_tr, X_val)
        #cart on different min_impurity_decrerase
        result_L = []
        for l in leafs:
            result_l = modules.cart(l, decrease, X_tr_norm, y_tr, X_val_norm, y_val)
            result_L.append(result_l)
        result.append(result_L)
mean_Etrain = np.mean(np.array(result)[:,:,5], axis = 0)
var_Etrain = np.var(np.array(result)[:,:,5], axis = 0)
mean_Eval = np.mean(np.array(result)[:,:,6], axis = 0)
var_Eval = np.var(np.array(result)[:,:,6], axis = 0)
mean_depth = np.mean(np.array(result)[:,:,0], axis = 0)
mean_leafs = np.mean(np.array(result)[:,:,1], axis = 0)
#one standard error of module:
#https://stats.stackexchange.com/questions/80268/empirical-justification-for-the-one-standard-error-rule-when-using-cross-validat
min_Eval = np.min(mean_Eval)
std_err = np.sqrt(var_Eval[np.argmin(mean_Eval)])/np.sqrt(T*k)
bound_Eval = min_Eval+std_err
simplest_module = 0
while(mean_Eval[simplest_module] > bound_Eval):
    simplest_module = simplest_module + 1
print("simplest module with leafs = " + str(leafs[simplest_module]))
print("Etrain = " + str(mean_Etrain[simplest_module]))
print("Eval = " + str(mean_Eval[simplest_module]))

plt.subplot(221)
plt.plot(leafs, mean_Etrain, 'b', label = 'mean_Etrain')
plt.plot(leafs, mean_Eval, 'r', label = 'mean_Eval')
plt.legend()
plt.subplot(222)
plt.plot(leafs, var_Etrain, 'b', label = 'var_Etrain')
plt.plot(leafs, var_Eval, 'r', label = 'var_Eval')
plt.legend()
plt.subplot(223)
plt.plot(leafs, mean_depth, label = 'mean_depth')
plt.legend()
plt.subplot(224)
plt.plot(leafs, mean_leafs, label = 'mean_leafs')
plt.legend()
plt.suptitle("CART with different max_leaf_nodes")
plt.show()
'''

# Random Forest: use the choosen tree from CART
decrease = 10**3.5
leafs = 34
N_trees=[50,100, 200, 300, 400, 500]
N_samples= [3,4,5,6,7,8]
T = 10
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True,random_state=None)
#save the result:
result = []
for t in range(T):
    for tr_index, v_index in skf.split(X, y):
        #training data
        X_tr = X[tr_index,:]
        y_tr = y[tr_index]
        #validation data
        X_val = X[v_index,:]
        y_val = y[v_index]
        #standardize
        (X_tr_norm, X_val_norm) = utils.normalizing(X_tr, X_val)
        #cart on different min_impurity_decrerase
        result_L = []
        for j in N_samples: 
            for i in N_trees:
                result_ij = modules.random_forest(i, j, leafs, decrease, X_tr_norm, y_tr, X_val_norm, y_val)
                result_L.append(result_ij)
        result.append(result_L)

mean_Etrain = np.mean(np.array(result)[:,:,3], axis = 0)
var_Etrain = np.var(np.array(result)[:,:,3], axis = 0)
mean_Eval = np.mean(np.array(result)[:,:,4], axis = 0)
var_Eval = np.var(np.array(result)[:,:,4], axis = 0)
best = np.argmin(mean_Eval)
print("----")
print(best)
print("mean_Etrain: " + str(mean_Etrain[best]))
print("mean_Etest: " + str(mean_Eval[best]))
print("var_Etrain: " + str(var_Etrain[best]))
print("var_Etest: " + str(var_Eval[best]))

mean_Etr = np.reshape(mean_Etrain, (6,6))
var_Etr = np.reshape(var_Etrain, (6,6))
mean_Ete = np.reshape(mean_Eval, (6,6))
var_Ete = np.reshape(var_Eval, (6,6))


plt.figure(0)
for i in range(3):
    for j in range(2):
        ax = plt.subplot2grid((3,2), (i,j))
        if j == 0:
            ax.plot(N_trees, mean_Etr[:,i].reshape(-1,1), label='mean_Etrain')
            ax.plot(N_trees, mean_Ete[:,i].reshape(-1,1), label='mean_Eval')
            plt.legend()
        else:
            ax.plot(N_trees, var_Etr[:,i].reshape(-1,1), label='var_Etrain')
            ax.plot(N_trees, var_Ete[:,i].reshape(-1,1), label='var_Eval')
            plt.legend()
plt.show()

plt.figure(1)
for i in range(3,6):
    for j in range(2):
        ax = plt.subplot2grid((3,2), (i-3,j))
        if j == 0:
            ax.plot(N_trees, mean_Etr[:,i].reshape(-1,1), label='mean_Etrain')
            ax.plot(N_trees, mean_Ete[:,i].reshape(-1,1), label='mean_Eval')
            plt.legend()
        else:
            ax.plot(N_trees, var_Etr[:,i].reshape(-1,1), label='var_Etrain')
            ax.plot(N_trees, var_Ete[:,i].reshape(-1,1), label='var_Eval')
            plt.legend()
plt.show()


# Gradiant Boosting Regression:
N_estimate=[10, 100, 300, 400, 500, 600, 700, 800, 2000]
N_depth=1
tol = 1e-4
T = 10
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True,random_state=None)
#save the result:
result = []
for t in range(T):
    for tr_index, v_index in skf.split(X, y):
        #training data
        X_tr = X[tr_index,:]
        y_tr = y[tr_index]
        #validation data
        X_val = X[v_index,:]
        y_val = y[v_index]
        #standardize
        (X_tr_norm, X_val_norm) = utils.normalizing(X_tr, X_val)
        #cart on different min_impurity_decrerase
        result_L = []
        for n in N_estimate:
            result_n = modules.GDboosting(n, N_depth, tol, X_tr_norm, y_tr, X_val_norm, y_val)
            result_L.append(result_n)
        result.append(result_L)

mean_Etrain = np.mean(np.array(result)[:,:,3], axis = 0)
var_Etrain = np.var(np.array(result)[:,:,3], axis = 0)
mean_Eval = np.mean(np.array(result)[:,:,4], axis = 0)
var_Eval = np.var(np.array(result)[:,:,4], axis = 0)
# one standard error
min_Eval = np.min(mean_Eval)
std_err = np.sqrt(var_Eval[np.argmin(mean_Eval)])/np.sqrt(T*k)
bound_Eval = min_Eval+std_err
simplest_module = 0
while(mean_Eval[simplest_module] > bound_Eval):
    simplest_module = simplest_module + 1
print("simplest module with boosting = " + str(N_estimate[simplest_module]))
print("mean_Etrain = " + str(mean_Etrain[simplest_module]))
print("mean_Eval = " + str(mean_Eval[simplest_module]))
print("var_Etrain = " + str(var_Etrain[simplest_module]))
print("var_Eval = " + str(var_Eval[simplest_module]))

plt.subplot(121)
plt.plot(np.log10(N_estimate), mean_Etrain, label='mean_Etrain')
plt.plot(np.log10(N_estimate), mean_Eval, label='mean_Eval')
plt.legend()
plt.subplot(122)
plt.plot(np.log10(N_estimate), var_Etrain, label='var_Etrain')
plt.plot(np.log10(N_estimate), var_Eval, label='var_Eval')
plt.legend()
plt.show()




'''
--------------------------------------------------------------------
Using training dataset to choose the best module by comparing between different modules.
'''
# using cross-validation to compare between different modules:
T = 10
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True,random_state=None)
#save the result:
result_const = []
result_MLE = []
result_MLE_pca = []
result_ridge = []
result_lasso = []
result_knn_uniform = []
result_knn_dist = []
result_CART = []
result_RF = []
result_boosting = []
for t in range(T):
    for tr_index, v_index in skf.split(X, y):
        #training data
        X_tr = X[tr_index,:]
        y_tr = y[tr_index]
        #validation data
        X_val = X[v_index,:]
        y_val = y[v_index]

        #####Constant module#####
        result0 = modules.constant(X_tr, y_tr, X_val, y_val)
        result_const.append(result0)

        #####Linear Regression Module#####
        #standardize
        (X_tr_std, X_val_std) = utils.standardizing(X_tr, X_val)
        #MLE on all features[baseline]
        result1 = modules.MLE(X_tr_std, y_tr, X_val_std, y_val)
        result_MLE.append(result1)
        #MLE with pca = 7
        c = 7
        result2 = modules.MLE_pca(c, X_tr_std, y_tr, X_val_std, y_val)
        result_MLE_pca.append(result2)
        #Ridge Regression with lambda = 25
        l1 = 25
        result3 = modules.ridge(l1, X_tr_std, y_tr, X_val_std, y_val)
        result_ridge.append(result3)
        #Lasso Regression with lambda = 35
        l2 = 35
        result4 = modules.lasso(l2, X_tr_std, y_tr, X_val_std, y_val)
        result_lasso.append(result4)


        #####Other Regreession Module#####
        #normalize
        (X_tr_norm, X_val_norm) = utils.normalizing(X_tr, X_val)
        #KNN (uniform) with k = 3
        k1 = 3
        w1 = 'uniform'
        result5 = modules.knn(k1, w1, X_tr_norm, y_tr, X_val_norm, y_val)
        result_knn_uniform.append(result5)
        #KNN (distance) with k = 4
        k2 = 4
        w2 = 'distance'
        result6 = modules.knn(k2, w2, X_tr_norm, y_tr, X_val_norm, y_val)
        result_knn_dist.append(result6)
        #CART with max_leaf_nodes = 34, min_impurity_decrease = 10^3.5
        leafs = 34
        decrease = 10**3.5
        result7 = modules.cart(leafs, decrease, X_tr_norm, y_tr, X_val_norm, y_val)
        result_CART.append(result7)
        #Random Forest with N_trees = 500, N_samples = 6
        trees = 500
        samples = 6
        result8 = modules.random_forest(trees, samples, leafs, decrease, X_tr_norm, y_tr, X_val_norm, y_val)
        result_RF.append(result8)
        #Gradian Boosting Regression with boosting = 400, depth = 1
        boosting = 400
        depth = 1
        tol = 1e-4
        result9 = modules.GDboosting(boosting, depth, tol, X_tr_norm, y_tr, X_val_norm, y_val)
        result_boosting.append(result9)

#Constant Model[baseline]
print("-------")
print("Constant Model: ")
print(" Etrain[mean var] " + str(np.mean(np.array(result_const)[:,1])) + " " + str(np.var(np.array(result_const)[:,1])))
print(" Eval[mean var] " + str(np.mean(np.array(result_const)[:,2])) + " " + str(np.var(np.array(result_const)[:,2])))
print(" constant of best Eval: " )
print(np.array(result_const)[np.argmin(np.array(result_const)[:,2])][0])
#MLE on all features[baseline]
print("-------")
print("MLE on all features: ")
print(" Etrain[mean var] " + str(np.mean(np.array(result_MLE)[:,3])) + " " + str(np.var(np.array(result_MLE)[:,3])))
print(" Eval[mean var] " + str(np.mean(np.array(result_MLE)[:,4])) + " " + str(np.var(np.array(result_MLE)[:,4])))
print(" coef of best Eval: " )
print(np.array(result_MLE)[np.argmin(np.array(result_MLE)[:,4])][0])
#MLE with pca = 7
print("-------")
print("MLE with PCA (c = 7): ")
print(" Etrain[mean var] " + str(np.mean(np.array(result_MLE_pca)[:,4])) + " " + str(np.var(np.array(result_MLE_pca)[:,4])))
print(" Eval[mean var] " + str(np.mean(np.array(result_MLE_pca)[:,5])) + " " + str(np.var(np.array(result_MLE_pca)[:,5])))
print(" coef of best Eval: " )
print(np.array(result_MLE_pca)[np.argmin(np.array(result_MLE_pca)[:,5])][1])
print(" pca_ration: ")
print(np.array(result_MLE_pca)[np.argmin(np.array(result_MLE_pca)[:,5])][0])
#Ridge Regression with lambda = 25
print("-------")
print("Ridge Regression with lambda = 25: ")
print(" Etrain[mean var] " + str(np.mean(np.array(result_ridge)[:,3])) + " " + str(np.var(np.array(result_ridge)[:,3])))
print(" Eval[mean var] " + str(np.mean(np.array(result_ridge)[:,4])) + " " + str(np.var(np.array(result_ridge)[:,4])))
print(" coef of best Eval: " )
print(np.array(result_ridge)[np.argmin(np.array(result_ridge)[:,4])][0])
#Lasso Regression with lambda = 35
print("-------")
print("Lasso Regression with lambda = 35: ")
print(" Etrain[mean var] " + str(np.mean(np.array(result_lasso)[:,3])) + " " + str(np.var(np.array(result_lasso)[:,3])))
print(" Eval[mean var] " + str(np.mean(np.array(result_lasso)[:,4])) + " " + str(np.var(np.array(result_lasso)[:,4])))
print(" coef of best Eval: " )
print(np.array(result_lasso)[np.argmin(np.array(result_lasso)[:,4])][0])
#KNN (uniform) with k = 3
print("-------")
print("KNN (uniform) with k = 3: ")
print(" Etrain[mean var] " + str(np.mean(np.array(result_knn_uniform)[:,2])) + " " + str(np.var(np.array(result_knn_uniform)[:,2])))
print(" Eval[mean var] " + str(np.mean(np.array(result_knn_uniform)[:,3])) + " " + str(np.var(np.array(result_knn_uniform)[:,3])))
#KNN (distance) with k = 4
print("-------")
print("KNN (distance) with k = 4: ")
print(" Etrain[mean var] " + str(np.mean(np.array(result_knn_dist)[:,2])) + " " + str(np.var(np.array(result_knn_dist)[:,2])))
print(" Eval[mean var] " + str(np.mean(np.array(result_knn_dist)[:,3])) + " " + str(np.var(np.array(result_knn_dist)[:,3])))
#CART with max_leaf_nodes = 34, min_impurity_decrease = 10^3.5
print("-------")
print("CART with max_leaf_nodes = 34, min_impurity_decrease = 10^3.5: ")
print(" Etrain[mean var] " + str(np.mean(np.array(result_CART)[:,5])) + " " + str(np.var(np.array(result_CART)[:,5])))
print(" Eval[mean var] " + str(np.mean(np.array(result_CART)[:,6])) + " " + str(np.var(np.array(result_CART)[:,6])))
print(" feature importance of best Eval: " )
print(np.array(result_CART)[np.argmin(np.array(result_CART)[:,6])][2])
#Random Forest with N_trees = 500, N_samples = 6
print("-------")
print("Random Forest with N_trees = 500, N_samples = 6: ")
print(" Etrain[mean var] " + str(np.mean(np.array(result_RF)[:,3])) + " " + str(np.var(np.array(result_RF)[:,3])))
print(" Eval[mean var] " + str(np.mean(np.array(result_RF)[:,4])) + " " + str(np.var(np.array(result_RF)[:,4])))
print(" feature importance of best Eval: " )
print(np.array(result_RF)[np.argmin(np.array(result_RF)[:,4])][0])
#Gradian Boosting Regression with boosting = 400, depth = 1
print("-------")
print("Gradian Boosting Regression with boosting = 400, depth = 1: ")
print(" Etrain[mean var] " + str(np.mean(np.array(result_boosting)[:,3])) + " " + str(np.var(np.array(result_boosting)[:,3])))
print(" Eval[mean var] " + str(np.mean(np.array(result_boosting)[:,4])) + " " + str(np.var(np.array(result_boosting)[:,4])))
print(" feature importance of best Eval: " )
print(np.array(result_boosting)[np.argmin(np.array(result_boosting)[:,4])][0])



'''
------------------------------------------------------------
choose the best model, train it with whole training datae and save the trained model
'''

# train the final module with whole training data 
(X_test, y_test) = utils.load("/Users/mac-pro/Desktop/20Fall/EE660/HW/Final/code/testing.csv")
(X_training_norm, X_test_norm) = utils.normalizing(X, X_test)

boosting = 400
depth = 1
tol = 1e-4
model = GradientBoostingRegressor(loss='ls', n_estimators = boosting, max_depth = depth, tol = tol, subsample = 1.0)
model.fit(X_training_norm,y)

# save the model to disk
# https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
filename = '/Users/mac-pro/Desktop/20Fall/EE660/HW/Final/code/final_model.sav'
pickle.dump(model, open(filename, 'wb'))
