#Regression models with randomly creeated data.
from sklearn import linear_model, metrics
import matplotlib.pyplot as plt
import numpy as np
import random

    

def create_y_data(variance):
    y_data = []
    for x in range(1000):
        y_data.append(x + (random.randint(0,variance)))
    return y_data 

def ridge_regression(y_data, variance):
    x_data = []
    for x in range(1000):
        x_data.append(x)
    x_data = np.array(x_data).transpose()
    y_data = np.array(y_data).transpose()
    y_train = y_data[:-200]
    y_test =  y_data[-200:]
    x_train = x_data[:-200]
    x_test = x_data[-200:]
    
    
    
    ridgereg = linear_model.Ridge()
    ridgereg.fit(x_train.reshape(-1,1), y_train)
    y_prediction = ridgereg.predict(x_test.reshape(-1,1))
    MSE = metrics.mean_squared_error(y_test,y_prediction)
    r2score = metrics.r2_score(y_test,y_prediction)
    
    
    
    plt.scatter(x_test, y_test, color = "black")
    plt.plot(x_test, y_prediction, color = "red")
    plt.ylabel("Output")
    plt.xlabel("Input")
    plt.title("Ridge Regression, Rnd Y Change = %s" %variance)
    plt.show()
    print("Mean Squared Error = %.2f" %MSE)
    print("R2 Score = %.2f" %r2score)


ridge_regression(create_y_data(1), "1")

ridge_regression(create_y_data(10), "10")

ridge_regression(create_y_data(50), "50")

ridge_regression(create_y_data(100), "100")