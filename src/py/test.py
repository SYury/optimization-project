import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from torch.optim.lbfgs import LBFGS
from a2grad import A2Grad

class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)

class RegressionTest:
    def __init__(self, name, model, create_opt, criterion, x, y):
        self.name = name
        self.model = model
        self.opt = create_opt(model)
        self.criterion = criterion
        self.x = x
        self.y = y
        self.errors = []
    
    def run(self, n_epochs):
        x_v = Variable(torch.from_numpy(self.x))
        y_v = Variable(torch.from_numpy(self.y))
        for i in range(n_epochs):
            self.opt.zero_grad()
            y_pred = self.model(x_v)
            loss = self.criterion(y_pred, y_v)
            loss.backward()
            self.opt.step(lambda : 0)
            self.errors.append(loss.data.item())
    
    def output(self, epochs):
        s = self.name + ' '
        for e in epochs:
            s += "{} ".format(self.errors[e])
        print(s)

class RegressionAnswer:
    def __init__(self, model, create_opt, criterion, x, y):
        self.model = model
        self.opt = create_opt(model)
        self.criterion = criterion
        self.x = x
        self.y = y
        self.answer = 0
    
    def run(self, duration):
        start_time = time.time()
        time.clock()
        elapsed = 0
        x_v = Variable(torch.from_numpy(self.x))
        y_v = Variable(torch.from_numpy(self.y))
        while elapsed < duration:
            elapsed = time.time() - start_time
            self.opt.zero_grad()
            y_pred = self.model(x_v)
            loss = self.criterion(y_pred, y_v)
            loss.backward()
            self.opt.step(lambda : 0)
            self.answer = loss.data.item()
        return self.answer

def test_linreg(n_epochs):
    print('Test: linear regression')
    
    x_values = []
    y_values = []
    
    for i in range(5):
        x_values.append(i)
        y_values.append(5*i + 2 + torch.randn(1).data.item())
    
    x_data = np.array(x_values, dtype=np.float32).reshape(-1, 1)
    y_data = np.array(y_values, dtype=np.float32).reshape(-1, 1)
    
    answer = RegressionAnswer(LinearRegression(), lambda model : LBFGS(model.parameters()), torch.nn.MSELoss(), x_data, y_data).run(60)
    
    test = [
        RegressionTest('A2Grad-uni', LinearRegression(), lambda model : A2Grad(model.parameters(), 'uni', 1e-1), torch.nn.MSELoss(), x_data, y_data), 
        RegressionTest('A2Grad-inc', LinearRegression(), lambda model : A2Grad(model.parameters(), 'inc', 1e-1), torch.nn.MSELoss(), x_data, y_data),
        RegressionTest('A2Grad-exp', LinearRegression(), lambda model : A2Grad(model.parameters(), 'exp', 1e-1), torch.nn.MSELoss(), x_data, y_data),
        RegressionTest('Adam', LinearRegression(), lambda model : Adam(model.parameters()), torch.nn.MSELoss(), x_data, y_data),
        RegressionTest('SGD', LinearRegression(), lambda model : SGD(model.parameters(), lr=1e-2), torch.nn.MSELoss(), x_data, y_data),
        RegressionTest('LBFGS', LinearRegression(), lambda model : LBFGS(model.parameters()), torch.nn.MSELoss(), x_data, y_data)
        ]
    
    plt.figure(figsize=(14, 8))
    for i in range(len(test)):
        test[i].run(n_epochs)
        plt.plot(np.arange(1, n_epochs + 1), np.array(test[i].errors) - answer, label=test[i].name)
    plt.legend(fontsize=12, loc=1)
    plt.title('Linear regression')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.savefig('linear.png')
    
    plt.figure(figsize=(14, 8))
    for i in range(len(test)):
        plt.plot(np.arange(1, n_epochs + 1), np.array(test[i].errors) - answer, label=test[i].name)
    plt.legend(fontsize=12, loc=1)
    plt.ylim(0, 1e-5)
    plt.title('Linear regression')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.savefig('linear2.png')
    
    points = np.arange(10, n_epochs, 10)
    header = "method   "
    for i in points:
        header += "{}                   ".format(i)
    print(header)
    for i in range(len(test)):
        test[i].output(points)
    print('')

class LogisticRegression(torch.nn.Module):
     def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
     def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

def test_logreg(n_epochs):
    print('Test: logistic regression')
    
    x_values = []
    y_values = []
    
    for i in range(5):
        x_values.append(i)
        y_values.append(5*i + 2 + torch.randn(1).data.item())
    
    x_data = np.array(x_values, dtype=np.float32).reshape(-1, 1)
    y_data = np.array(y_values, dtype=np.float32).reshape(-1, 1)
    
    answer = RegressionAnswer(LogisticRegression(), lambda model : LBFGS(model.parameters()), torch.nn.MSELoss(), x_data, y_data).run(60)
    
    test = [
        RegressionTest('A2Grad-uni', LogisticRegression(), lambda model : A2Grad(model.parameters(), 'uni', 1e-1), torch.nn.MSELoss(), x_data, y_data), 
        RegressionTest('A2Grad-inc', LogisticRegression(), lambda model : A2Grad(model.parameters(), 'inc', 1e-1), torch.nn.MSELoss(), x_data, y_data),
        RegressionTest('A2Grad-exp', LogisticRegression(), lambda model : A2Grad(model.parameters(), 'exp', 1e-1), torch.nn.MSELoss(), x_data, y_data),
        RegressionTest('Adam', LogisticRegression(), lambda model : Adam(model.parameters()), torch.nn.MSELoss(), x_data, y_data),
        RegressionTest('SGD', LogisticRegression(), lambda model : SGD(model.parameters(), lr=1e-2), torch.nn.MSELoss(), x_data, y_data),
        RegressionTest('LBFGS', LogisticRegression(), lambda model : LBFGS(model.parameters()), torch.nn.MSELoss(), x_data, y_data)
        ]
    
    plt.figure(figsize=(14, 8))
    for i in range(len(test)):
        test[i].run(n_epochs)
        plt.plot(np.arange(1, n_epochs + 1), np.array(test[i].errors) - answer, label=test[i].name)
    plt.legend(fontsize=12, loc=1)
    plt.title('Logistic regression')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.savefig('logistic.png')
    
    plt.figure(figsize=(14, 8))
    for i in range(len(test)):
        plt.plot(np.arange(1, n_epochs + 1), np.array(test[i].errors) - answer, label=test[i].name)
    plt.legend(fontsize=12, loc=1)
    plt.title('Logistic regression')
    plt.ylim(0, 1e-3)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.savefig('logistic2.png')
    
    points = np.arange(10, n_epochs, 10)
    header = "method   "
    for i in points:
        header += "{}                   ".format(i)
    print(header)
    for i in range(len(test)):
        test[i].output(points)
    print('')


def main():
    test_linreg(1002)
    test_logreg(2002)

main()
