import numpy as np


""" Logistic regression using Gradient descent method"""
class LogisticRegressionG():


  # learning rate & number of iterations (parameters of the algorithm)
  def __init__(self, learning_rate, number_of_iterations):

    self.learning_rate = learning_rate
    self.number_of_iterations = number_of_iterations


  # Gradient descent implementation: 
  def update_theta(self):

    # Hypothesis (sigmoid function)

    h = 1 / (1 + np.exp( - (self.X.dot(self.w) + self.b ) ))


    # derivaties

    dw = (1/self.m)*np.dot(self.X.T, (h - self.Y))

    db = (1/self.m)*np.sum(h - self.Y)


    # updating theta(w and b) using gradient descent

    self.w = self.w - self.learning_rate * dw

    self.b = self.b - self.learning_rate * db
    
  # the fit function for training the model
  def fit(self, X, Y):

    # m: number of rows
    # n: nul=mber of columns
    self.m, self.n = X.shape


    #initiating weight & bias value (theta)

    self.w = np.zeros(self.n)

    self.b = 0

    self.X = X

    self.Y = Y


    # Using of Gradient Descent for Optimization of the cost function

    for i in range(self.number_of_iterations):
      self.update_theta()


  # Sigmoid Equation & prediction function

  def predict(self, X):

    Y_pred = 1 / (1 + np.exp( - (X.dot(self.w) + self.b ) ))
    Y_pred = np.where( Y_pred > 0.5, 1, 0)
    return Y_pred





""" Logistic regression using Newton's method"""

class LogisticRegressionN:
    def __init__(self, learning_rate, number_of_iterations):
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
    
    # the segmoid fucntion

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
   
   # the hessian matrix of X
   # H= XT*diag(h(1-h))*X

    def hessian(self, X, h):
        return np.dot(X.T, np.dot(np.diag(h * (1 - h)), X))
    
   # Implementation of Newton's method

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0

        for i in range(self.number_of_iterations):
            z = np.dot(X, self.w) + self.b
            h = self.sigmoid(z)
            gradient_w = (1 / self.m) * np.dot(X.T, (h - y))
            gradient_b = (1 / self.m) * np.sum(h - y)
            H = self.hessian(X, h)
            H_inv = np.linalg.inv(H)
            self.w -= self.learning_rate * np.dot(H_inv, gradient_w)
            self.b -= self.learning_rate * gradient_b
            
    # The prediction function

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        y_pred = self.sigmoid(z)
        y_pred = np.where(y_pred > 0.5, 1, 0)
        return y_pred
