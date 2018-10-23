import numpy as np
np.random.seed(456)
import deepchem as dc

def gen_cls_data(n=100):
    x_zeros = np.random.multivariate_normal(mean=np.array((-1,-1)), cov=.1*np.eye(2), size=(n//2,))
    y_zeros = np.zeros((n//2,))
    
    x_ones = np.random.multivariate_normal(mean=np.array((1,1)), cov=.1*np.eye(2), size=(n//2,))
    y_ones = np.ones((n//2,))
    
    x_np = np.vstack([x_zeros, x_ones])
    y_np = np.concatenate([y_zeros, y_ones])
    return (x_np, y_np)


def gen_reg_data(w, b, n=50, d=1, noise_scale=.1):
    x_np = np.random.rand(n, d)
    noise = np.random.normal(scale=nose_scale, size=(n,d))
    y_np = np.reshape(w * x_np + b + noise, (-1))
    return (x_np, y_np)

def tox21():
    _, (train, valid, test), _ = dc.molnet.load_tox21()
    train_X, train_y, train_w = train.X, train.y, train.w
    valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
    test_X, test_y, test_w = test.X, test.y, test.w
    
    train_y = train_y[:, 0]
    valid_y = valid_y[:, 0]
    test_y = test_y[:, 0]
    train_w = train_w[:, 0]
    valid_w = valid_w[:, 0]
    test_w = test_w[:, 0]
    
    return (train_X, train_y, train_w, valid_X, valid_y, valid_w, test_X, test_y, test_w)