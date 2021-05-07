import numpy as np
import utils
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

(X_training, y_training) = utils.load("/Users/mac-pro/Desktop/20Fall/EE660/HW/Final/code/training.csv")
(X_test, y_test) = utils.load("/Users/mac-pro/Desktop/20Fall/EE660/HW/Final/code/testing.csv")
(X_training_norm, X_test_norm) = utils.normalizing(X_training, X_test)

#load trained model
#https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
filename = '/Users/mac-pro/Desktop/20Fall/EE660/HW/Final/code/final_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
y_training_pred = loaded_model.predict(X_training_norm)
y_test_pred = loaded_model.predict(X_test_norm)
error_training = np.sqrt(mean_squared_error(y_training, y_training_pred))
error_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("Final result with Gradient boosting model:")
print("RMSE on training dataset = " + str(error_training))
print("RMSE on test dataset = " + str(error_test))
print("Features importance: ")
print(loaded_model.feature_importances_)
