import numpy as np

def RMSE(preds, truth):
    return np.sqrt(np.mean(np.square(preds-truth)))

def RMSE_label(preds, truth,label):
    lista =[]
    for i in range(len(preds)):
        lista.append(label[preds[i]])
    return np.sqrt(np.mean(np.square(lista-truth)))