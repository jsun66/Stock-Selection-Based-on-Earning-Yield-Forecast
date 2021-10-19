# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import Dropout

# Fix the random number seed to ensure reproducibility
np.random.seed(7)

# Importing the dataset
X = pd.read_csv('X2.csv')
y = pd.read_csv('y2.csv')
X = X.iloc[:,3:].values
y = y.iloc[:,3].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
sc_X = StandardScaler()
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))
y_test = sc_y.transform(y_test.reshape(-1,1))


# Part 2 - Making the ANN

# Initialising the ANN
regressor = Sequential()

# Adding the input layer and the first hidden layer
regressor.add(Dense(units = 58, kernel_initializer = 'he_uniform', activation = 'selu', input_dim = 112))
#regressor.add(Dropout(0.1))

# Adding the second hidden layer
regressor.add(Dense(units = 58, kernel_initializer = 'he_uniform', activation = 'selu'))
#regressor.add(Dropout(0.1))

# Adding the third hidden layer
regressor.add(Dense(units = 58, kernel_initializer = 'he_uniform', activation = 'selu'))
#regressor.add(Dropout(0.1))

# Adding the output layer
regressor.add(Dense(units = 1, kernel_initializer = 'he_uniform', activation = 'selu'))

# Compiling the ANN
#from keras.optimizers import Adam
#optimizer = Adam(lr=0.3)
regressor.compile(optimizer = 'adam', loss = 	'mean_squared_logarithmic_error', metrics = ['mean_squared_logarithmic_error'])

# Fitting the ANN to the Training set
history = regressor.fit(X_train, y_train, batch_size = 100, epochs = 4500)

# Plot Metrics
from matplotlib import pyplot
pyplot.plot(history.history['mean_squared_logarithmic_error'])
pyplot.show()

# Evaluate the model
#loss_t = regressor.evaluate(X_train, y_train)
#print("\nLoss: %.5f" % (loss_t))
#loss_tt = regressor.evaluate(X_test, y_test)
#print("\nLoss: %.5f" % (loss_tt)) 


# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = regressor.predict(X_test)
#y_pred = (y_pred > 0.5)
y_predreal=sc_y.inverse_transform(y_pred)
y_testreal=sc_y.inverse_transform(y_test)
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import metrics
rmse = sqrt(mean_squared_error(y_predreal, y_testreal))
r2 = metrics.r2_score(y_predreal, y_testreal)
error = sorted((y_predreal-y_testreal)/y_testreal)
#error = pd.DataFrame(error)

new_prediction = regressor.predict(sc_X.transform(X))
pred  = pd.DataFrame(new_prediction)
pred.to_csv('C:/Users/daphn/Desktop/sim/pred.csv', index = False)
pred = pd.read_csv('pred.csv')
#e_max = np.max(error)
#e_min = np.min(error)
#import pylab as pl
#import scipy.stats as stats
#fit = stats.norm.pdf(error, np.mean(error), np.std(error))
#pl.plot(error,fit,'-o')
#pl.hist(error,normed=True)
#pl.show()
#
#error.sort()
#e_mean = np.mean(error)
#e_std = np.std(error)
#pdf = stats.norm.pdf(error, e_mean, e_std)
#pyplot.plot(error, pdf)
#
#import seaborn as sns
#ax = sns.distplot(error,
#                  bins=100,
#                  kde=False,
#                  color='skyblue',
#                  hist_kws={"linewidth": 15,'alpha':1})
#ax.set(xlabel='Normal', ylabel='Frequency')


#pyplot.scatter(y_testreal, color = 'red', label = 'Real EBIT')
#pyplot.scatter(y_predreal, color = 'blue', label = 'Predicted EBIT')
#pyplot.title('EBIT Prediction')
#pyplot.xlabel('Companies')
#pyplot.ylabel('EBIT')
#pyplot.legend()
#pyplot.show()

# Predicting a single new observation
new_prediction = regressor.predict(sc_X.transform(np.array([[0, 0, ]])))
#new_prediction = (new_prediction > 0.5)

# Part 4 - Evaluating, Improving and Tuning the ANN

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
#from keras.layers import Dropout
def build_regressor():
    regressor = Sequential()
    regressor.add(Dense(units = 1024, activation = 'relu', input_dim = 56))
    #regressor.add(Dropout(0.1))
    regressor.add(Dense(units = 1024, activation = 'relu'))
    #regressor.add(Dropout(0.1))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse'])
    return regressor
regressor = KerasRegressor(build_fn = build_regressor, batch_size = 32, epochs = 100)
mse = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = mse.mean()
variance = mse.std()
#predictions = cross_val_predict(regressor, df, y, cv=10)

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed
#from keras.layers import Dropout

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
def build_regressor(optimizer):
    regressor = Sequential()
    regressor.add(Dense(units = 28, activation = 'relu', input_dim = 56))
    #regressor.add(Dropout(0.1))
    regressor.add(Dense(units = 28, activation = 'relu'))
    #regressor.add(Dropout(0.1))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['mse'])
    return regressor
regressor = KerasRegressor(build_fn = build_regressor, dropout_rate = 0.1)
parameters = {'batch_size': [25, 32],
              'epochs': [150, 200],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = regressor, param_grid = parameters, cv = 10, n_jobs = -1, scoring = 'mse')
grid_search = grid_search.fit(X_train, y_train)

# Summarize results
print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
params = grid_search.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))