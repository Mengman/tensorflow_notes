import numpy as np
import json

def load_data(path='imdb.npz', num_words=None, skip_top=0, maxlen=None, seed=113, start_char=1, oov_char=2, index_from=3, **kwargs):
    with np.load(path) as f:
        x_train, labels_train = f['x_train'], f['y_train']
        x_test, labels_test = f['x_test'], f['y_test']
    
    np.random.seed(seed)
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    labels_train = labels_train[indices]
    
    indices = np.arange(len(x_test))
    np.random.shuffle(indices)
    x_test = x_test[indices]
    labels_test = labels_test[indices]
    
    xs = np.concatenate([x_train, x_test])
    labels = np.concatenate([labels_train, labels_test])
    
    if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        xs = [ [w + index_from for w in x] for x in xs ]
    
    if not num_words:
        num_words = max([max(x) for x in xs])
        
    if oov_char is not None:
        xs = [
            [w if (skip_top <= w < num_words) else oov_char for w in x ] for x in xs
        ]
    else:
        xs = [[w for w in x if skip_top <= w < num_words] for x in xs]
    
    idx = len(x_train)
    x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
    x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])
    return (x_train, y_train), (x_test, y_test)

def get_word_index(path="imdb_word_index.json"):
    with open(path) as f:
        return json.load(f)
