import gzip
import os
import numpy as np

def load_data():
    dirname = os.path.join("datasets", "fashion-mnist")
    files = [
      'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
      't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']
    with gzip.open(os.path.join(dirname, files[0]), 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    
    with gzip.open(os.path.join(dirname, files[1]), "rb") as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
        
    with gzip.open(os.path.join(dirname, files[2]), "rb") as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    
    with gzip.open(os.path.join(dirname, files[3]), "rb") as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
    return (x_train, y_train), (x_test, y_test)