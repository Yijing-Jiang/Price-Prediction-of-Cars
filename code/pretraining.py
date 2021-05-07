import numpy as np
import pandas as pd
import utils
import modules
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

'''
split the data and save into files
'''
#create csv files to save the data (delete missing data and convert catogary to numerical)
data_delete_missing = utils.csv_to_X_and_y('/Users/mac-pro/Desktop/20Fall/EE660/HW/Final/code/cars_missing.csv')
#https://www.geeksforgeeks.org/convert-a-numpy-array-into-a-csv-file/
# convert array into dataframe 
DF = pd.DataFrame(data_delete_missing)
# save the dataframe as a csv file 
DF.to_csv("/Users/mac-pro/Desktop/20Fall/EE660/HW/Final/code/data_delete_missing.csv")

#divide pretraining, training, testing data
((D_pt, y_pt), (D_prime, y_prime), (D_test, y_test)) = utils.divide_D("/Users/mac-pro/Desktop/20Fall/EE660/HW/Final/code/data_delete_missing.csv")
#save pretrainig, training, testing dataset to csv files
pretraining = np.append(np.reshape(y_pt, (125,1)), D_pt, axis=1)
PT = pd.DataFrame(pretraining)
PT.to_csv("/Users/mac-pro/Desktop/20Fall/EE660/HW/Final/code/pretraining.csv") #pretraining
training = np.append(np.reshape(y_prime, (893,1)), D_prime, axis=1)
TR = pd.DataFrame(training)
TR.to_csv("/Users/mac-pro/Desktop/20Fall/EE660/HW/Final/code/training.csv") #training
testing = np.append(np.reshape(y_test, (224,1)), D_test, axis=1)
TE = pd.DataFrame(testing)
TE.to_csv("/Users/mac-pro/Desktop/20Fall/EE660/HW/Final/code/testing.csv") #testing



'''
--------------------------------------------------------------------
Using pre-training dataset to look into the data and try models.
'''
# load pretraining data
(D_pt, y_pt) = utils.load("/Users/mac-pro/Desktop/20Fall/EE660/HW/Final/code/pretraining.csv")

# view the data in standardize form (for LR)
(D_pt, D) = utils.standardizing(D_pt, D_pt)
plt.subplot(331)
plt.scatter(D_pt[:,0], y_pt)
plt.title('age')
plt.subplot(332)
plt.scatter(D_pt[:,1], y_pt)
plt.title('km')
plt.subplot(333)
plt.scatter(D_pt[:,2], y_pt)
plt.title('Fuel Type')
plt.subplot(334)
plt.scatter(D_pt[:,3], y_pt)
plt.title('Engine Power')
plt.subplot(335)
plt.scatter(D_pt[:,4], y_pt)
plt.title('MetColor')
plt.subplot(336)
plt.scatter(D_pt[:,5], y_pt)
plt.title('Gear type')
plt.subplot(337)
plt.scatter(D_pt[:,6], y_pt)
plt.title('Engine CC')
plt.subplot(338)
plt.scatter(D_pt[:,7], y_pt)
plt.title('Doors')
plt.subplot(339)
plt.scatter(D_pt[:,8], y_pt)
plt.title('weight')
plt.show()

#splite the pretraining data into training and test set
(Dpt_tr, Dpt_test, ypt_tr, ypt_test) = train_test_split(D_pt, y_pt, test_size = 0.1)




'''
----------------------------------------------------------
using linear regression related model: all need to standardize the data
'''

#standardize pretraining data for linear model
(Dpt_tr_std, Dpt_test_std) = utils.standardizing(Dpt_tr, Dpt_test)

#MLE baseline with all features:
(coef, y_train_pred, y_test_pred, error_train, error_test) = modules.MLE(Dpt_tr_std, ypt_tr, Dpt_test_std, ypt_test)
print("MLE")
print(coef, error_train, error_test)
# plot result with feature 0
x= np.array([-2, -1, 0, 1, 2])
y_pred = coef[0][0] * x + coef[1]
plt.plot(x,y_pred, 'r')
plt.scatter(Dpt_test_std[:,0].reshape(-1,1), ypt_test)
plt.show()

#MLE + one feature: feature 0 performs best
one_feature_error_train = []
one_feature_error_test = []
all_feature_error_train = []
all_feature_error_test = []
for i in range(0,9):
    result = modules.MLE(Dpt_tr_std[:,i].reshape(-1,1), ypt_tr, Dpt_test_std[:,i].reshape(-1,1), ypt_test)
    one_feature_error_train.append(result[3])
    one_feature_error_test.append(result[4])
    all_feature_error_train.append(error_train)
    all_feature_error_test.append(error_test)
print("MLE for one feature")
print(one_feature_error_train)
print(one_feature_error_test)

plt.plot(all_feature_error_train, 'r', label = 'E_train on MLE with all feature')
plt.plot(all_feature_error_test, 'b', label = 'E_test on MLE with all feature')
plt.plot(one_feature_error_train, 'y', label = 'E_train on MLE with one feature')
plt.plot(one_feature_error_test, 'g', label = 'E_test on MLE with one feature')
plt.legend()
plt.show()

#MLE + polynomial features: finally choose d = 2 for all features; 0 feature to plot
#all features: N_pretraining only has 125 data
for d in range(2,5):
    (Dpt_tr_std_poly, Dpt_test_std_poly) = utils.polynomial(d, Dpt_tr_std, Dpt_test_std)
    (coef, y_train_pred, y_test_pred, error_train, error_test) = modules.MLE(Dpt_tr_std_poly, ypt_tr, Dpt_test_std_poly, ypt_test)
    print("-----")
    print(d)
    print(coef, error_train, error_test)

#MLE + PCA: choose c = 7
etrain = []
etest = []
for c in range(1,9):
    (pca_ratio, Dpt_tr_std_pca, Dpt_test_std_pca) = utils.doPCA(c, Dpt_tr_std, Dpt_test_std)
    (coef, y_train_pred, y_test_pred, error_train, error_test) = modules.MLE(Dpt_tr_std_pca, ypt_tr, Dpt_test_std_pca, ypt_test)
    print("-------")
    print(c)
    print(coef, error_train, error_test)
    etrain.append(error_train)
    etest.append(error_test)

plt.plot([1,2,3,4,5,6,7,8], etrain, 'g', label = "train")
plt.plot([1,2,3,4,5,6,7,8], etest, 'b', label = "test")
plt.legend()
plt.show()



'''
#give some constraints on weights to less fit on the noise
# find a range of lamba: lambda larger, stronger regularization, weight will shrink more
# b is sparcity of weight. b smaller, more sparce, centered at (0,0) and axes
'''
# Ridge regression: l = [0.1, 1, 5, 10, 20, 40, 60, 80, 100, 150, 200] choose the best
L = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 30, 40, 50, 60, 100, 150, 200]
etrain = []
etest = []
for l in L:
    (coef, y_train_pred, y_test_pred, error_train, error_test) = modules.ridge(l, Dpt_tr_std, ypt_tr, Dpt_test_std, ypt_test)
    etrain.append(error_train)
    etest.append(error_test)

plt.plot(np.log10(L), etrain, 'g', label = "train")
plt.plot(np.log10(L), etest, 'b', label = "test")
plt.legend()
plt.show()

# Lasso regression: l = [1,...,1000]
L = [1, 5, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
etrain = []
etest = []
for l in L:
    (coef, y_train_pred, y_test_pred, error_train, error_test) = modules.lasso(l, Dpt_tr_std, ypt_tr, Dpt_test_std, ypt_test)
    etrain.append(error_train)
    etest.append(error_test)

plt.plot(np.log10(L), etrain, 'g', label = "train")
plt.plot(np.log10(L), etest, 'b', label = "test")
plt.legend()
plt.show()





'''
------------------------------------------------
nonlinear regression
'''
#normalize pretraining data for non linear model
(Dpt_tr_norm, Dpt_test_norm) = utils.normalizing(Dpt_tr, Dpt_test)

#Baseline: KNN: weights = ['uniform', 'distance'], k = 1 to 10
for k in range (1,10):
    (y_train_pred, y_test_pred, error_train, error_test) = modules.knn(k, 'distance', Dpt_tr_norm, ypt_tr, Dpt_test_norm, ypt_test)
    print("----")
    print(k)
    print(error_train, error_test)

'''
using ABM model (tree based)
'''
#CART: min_impurity_decrease = [0.001, 0.1, 1, 10, 100, 1000] and plot --- find not overfitting region
#       then try similar leafs around and plot, choose simplist model around +- 1 std Etest
min_impurity_decrease = 10
max_leaf_nodes = 90
result = modules.cart(max_leaf_nodes, min_impurity_decrease, Dpt_tr_norm, ypt_tr, Dpt_test_norm, ypt_test)
print("-----")
print(max_leaf_nodes)
print(result[0])
print(result[1])
print(result[2])
print(result[5:7])
plt.plot(np.arange(90,105,1), etrain, 'b', label = 'Etrain')
plt.plot(np.arange(90,105,1), etest, 'r', label = 'Etest')
plt.legend()
plt.show()

#RF: N_trees=[10, 50, 100, 200, 400, 800, 1000]; N_samples=[2,3,4,5,6,7,8]
# other parameters from best choice of CART
for d in range(3, 9): 
    result = modules.random_forest(100, d, 99, 10, Dpt_tr_norm, ypt_tr, Dpt_test_norm, ypt_test)
    print("------")
    print(d)
    print(result[2:4])

#GradiantBoostingRegression: N_boosting = [50, 100, 200, 300, 500, 800, 1000]; N_depth = [1,2,3,5]
N_boosting = 100
N_depth = 2
tol = 1e-4
result = modules.GDboosting(N_boosting, N_depth, tol, Dpt_tr_norm, ypt_tr, Dpt_test_norm, ypt_test)
print("----")
print(result[0])
print(result[3:5])

