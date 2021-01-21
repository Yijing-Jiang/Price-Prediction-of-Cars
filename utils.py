import numpy as np
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

#load data from original csv: delete missing data and change cate to num
def csv_to_X_and_y(filename):
    read = pd.read_csv(filename)
    csv_list = read.values.tolist()
    csv_array = np.zeros((1436,10))
    csv_array[:,0:3] = np.array([row[0:3] for row in csv_list])
    csv_array[:,4:10] = np.array([row[4:10] for row in csv_list])
    #delete missing data: nan in 'MetColor' feature
    missing_row_index = []
    for i in range(0,1436):
        if (csv_array[i][5] != 0 and csv_array[i][5] != 1) or not (csv_array[i][4]<200):
            missing_row_index.append(i)
        #define fuel type as Petrol-3, Diesel-2, other(CNG)-1
        if csv_list[i][3] == 'Petrol': csv_array[i][3] = 3
        elif csv_list[i][3] == 'Diesel': csv_array[i][3] = 2
        else: csv_array[i][3] = 1
    csv_array = np.delete(csv_array, np.array(missing_row_index), 0)
    #D = csv_array[:,1:]
    #y = csv_array[:,0]
    #(num_D, num_feature) = D.shape
    return csv_array

# divide data_delete_missing into pretraining, training and testing sets
def divide_D(filename):
    read = pd.read_csv(filename)
    csv_list = read.values.tolist()
    csv_array = np.array(csv_list)
    D = csv_array[:,2:]
    y = csv_array[:,1]
    #divide the dataset into pretraining, training and test set.
    D_model, D_pt, y_model, y_pt = train_test_split(D, y, test_size = 0.1)
    D_prime, D_test, y_prime, y_test = train_test_split(D_model, y_model, test_size = 0.2)
    return((D_pt, y_pt), (D_prime, y_prime), (D_test, y_test))

# load dataset from csv
def load(filename):
    read = pd.read_csv(filename)
    csv_list = read.values.tolist()
    csv_array = np.array(csv_list)
    D = csv_array[:,2:]
    y = csv_array[:,1]
    return (D, y)


#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
def normalizing(X_train, X_test):
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_normalized = scaler.transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    return(X_train_normalized, X_test_normalized)

#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
def standardizing(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_standard = scaler.transform(X_train)
    X_test_standard = scaler.transform(X_test)
    return(X_train_standard, X_test_standard)

#https://towardsdatascience.com/machine-learning-polynomial-regression-with-python-5328e4e8a386
#https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions
def polynomial(d, X_train, X_test):
    poly = PolynomialFeatures(degree = d)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.fit_transform(X_test)
    return(X_train_poly, X_test_poly)

# use PCA to transform the features(linear dimensionality reduction)
def doPCA(c, train, test):
    pca = PCA(n_components = c)
    pca.fit(train)
    train_PCA = pca.transform(train)
    test_PCA = pca.transform(test)
    return (pca.explained_variance_ratio_, train_PCA, test_PCA)