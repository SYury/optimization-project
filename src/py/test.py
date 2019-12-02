import torch
from torch.optim.adagrad import Adagrad
from torch.optim.lbfgs import LBFGS
from a2grad import A2Grad

methods = [
        'A2Grad-uni',
        'A2Grad-inc',
        'A2Grad-exp',
        'Adagrad', # быстрый градиентный метод
        'L-BFGS' # квазиньютоновский метод
]

def test_linreg():
    print('Test: linear regression')
    N = 3
    M = 10
    x = torch.randn((M, N))
    real_A = torch.randn((N, 1))
    y = x.mm(real_A) + torch.randn(M)
    A = torch.randn((N, 1), requires_grad=True)
    
    vec = [A, A, A, A, A]
    
    optimizer = [
        A2Grad([vec[0]], 'uni'),
        A2Grad([vec[1]], 'inc'),
        A2Grad([vec[2]], 'exp'),
        Adagrad([vec[3]]),
        LBFGS([vec[4]])
        ]

    errors = [[], [], [], [], []]
    
    points = [10, 20, 30, 40, 50]
    header = "method   "
    for i in points:
        header += "{}                   ".format(i)
    print(header)
    for n in range(1, 52):
        for i in range(len(methods)):
            optimizer[i].zero_grad()
            mse = torch.mean((x.mm(vec[i]) - y)**2)
            if n - 1 in points:
                errors[i].append(mse)
            mse.backward()
            optimizer[i].step(lambda : 0)
    
    for i in range(len(methods)):
        s = methods[i] + ' '
        for err in errors[i]:
            s += "{} ".format(err)
        print(s)
    print('')

def test_normsquare():
    print("Test: (1, 2, 3)^T*x + 2||x - (0, 1, 0)||^2")
    c = torch.tensor([1.0, 2.0, 3.0])
    xk = torch.tensor([0.0, 1.0, 0.0])
    x = torch.randn(3, requires_grad=True)

    vec = [x, x, x, x, x]
    
    optimizer = [
        A2Grad([vec[0]], 'uni'),
        A2Grad([vec[1]], 'inc'),
        A2Grad([vec[2]], 'exp'),
        Adagrad([vec[3]]),
        LBFGS([vec[4]])
        ]

    vals = [[], [], [], [], []]
    
    points = [10, 20, 30, 40, 50]
    header = "method   "
    for i in points:
        header += "{}                   ".format(i)
    print(header)
    for n in range(1, 52):
        for i in range(len(methods)):
            optimizer[i].zero_grad()
            val = torch.dot(c, vec[i]) + 2*torch.dist(vec[i], xk)**2
            if n - 1 in points:
                vals[i].append(val)
            val.backward()
            optimizer[i].step(lambda : 0)
    
    for i in range(len(methods)):
        s = methods[i] + ' '
        for val in vals[i]:
            s += "{} ".format(val)
        print(s)
    print('')

def main():
    test_linreg()
    test_normsquare()

main()
