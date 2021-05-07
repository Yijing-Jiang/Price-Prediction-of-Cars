import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from itertools import combinations
import utils
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# constant model:
def constant(X_train, y_train, X_test, y_test):
    y_pred = np.mean(y_train)
    y_train_pred = np.full(y_train.shape, y_pred)
    y_test_pred = np.full(y_test.shape, y_pred)
    error_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    error_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    return(y_pred, error_train, error_test)


# Baseline for LR: MLE
def MLE(X_train, y_train, X_test, y_test):
    MLE = LinearRegression()
    MLE.fit(X_train, y_train)
    coef = [MLE.coef_, MLE.intercept_] # coef is wi, intercept is w0
    y_train_pred = MLE.predict(X_train)
    y_test_pred = MLE.predict(X_test)
    error_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    error_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    return(coef, y_train_pred, y_test_pred, error_train, error_test)

# MLE + polynomial (d = 2)
def MLE_poly(d, X_train, y_train, X_test, y_test):
    (X_train_poly, X_test_poly) = utils.polynomial(d, X_train, X_test)
    (coef, y_train_pred, y_test_pred, error_train, error_test) = MLE(X_train_poly, y_train, X_test_poly, y_test)
    return(coef, y_train_pred, y_test_pred, error_train, error_test)

# MLE + PCA (c = 7)
def MLE_pca(c, X_train, y_train, X_test, y_test):
    (pca_ratio, X_train_pca, X_test_pca) = utils.doPCA(c, X_train, X_test)
    (coef, y_train_pred, y_test_pred, error_train, error_test) = MLE(X_train_pca, y_train, X_test_pca, y_test)
    return(pca_ratio, coef, y_train_pred, y_test_pred, error_train, error_test)
    
# Ridge regression (l is the lambda in course;)
def ridge(l, X_train, y_train, X_test, y_test):
    clf = Ridge(alpha=l)
    clf.fit(X_train, y_train)
    coef = [clf.coef_, clf.intercept_]
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    error_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    error_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    return(coef, y_train_pred, y_test_pred, error_train, error_test)

# Lasso regression (l is the lambda in course;)
def lasso(l, X_train, y_train, X_test, y_test):
    lasso = Lasso(alpha=l)
    lasso.fit(X_train, y_train)
    coef = [lasso.coef_, lasso.sparse_coef_, lasso.intercept_]
    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)
    error_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    error_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    return(coef, y_train_pred, y_test_pred, error_train, error_test)

# https://scikit-learn.org/stable/auto_examples/neighbors/plot_regression.html#sphx-glr-auto-examples-neighbors-plot-regression-py
def knn(k, weights, X_train, y_train, X_test, y_test):
    neigh = KNeighborsRegressor(n_neighbors=k, weights = weights)
    neigh.fit(X_train, y_train)
    y_train_pred = neigh.predict(X_train)
    y_test_pred = neigh.predict(X_test)
    error_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    error_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    return(y_train_pred, y_test_pred, error_train, error_test)

#https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor
#https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html#sphx-glr-auto-examples-tree-plot-tree-regression-py
def cart(max_leaf_nodes, min_impurity_decrease, X_train, y_train, X_test, y_test):
    tree = DecisionTreeRegressor(criterion="mse", splitter = "best", max_leaf_nodes = max_leaf_nodes, min_impurity_decrease=min_impurity_decrease)
    tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)
    error_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    error_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    return( tree.get_depth(), tree.get_n_leaves(), tree.feature_importances_, y_train_pred, y_test_pred, error_train, error_test)

#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
def random_forest(N_trees, N_samples, max_leaf_nodes, min_impurity_decrease, X_train, y_train, X_test, y_test):
    rf = RandomForestRegressor(n_estimators = N_trees, criterion = "mse", max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=True, random_state=None, max_samples = N_samples)
    rf.fit(X_train, y_train)
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)
    error_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    error_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    return(rf.feature_importances_, y_train_pred, y_test_pred, error_train, error_test)

def GDboosting(N_boosting, N_depth, tol, X_train, y_train, X_test, y_test):
    gdb = GradientBoostingRegressor(loss='ls', n_estimators = N_boosting, max_depth = N_depth, tol = tol, subsample = 1.0)
    gdb.fit(X_train,y_train)
    y_train_pred = gdb.predict(X_train)
    y_test_pred = gdb.predict(X_test)
    error_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    error_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    return(gdb.feature_importances_, y_train_pred, y_test_pred, error_train, error_test)

