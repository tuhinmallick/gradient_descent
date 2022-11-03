
import numpy as np
#Adagrad
#Mini Batch Adagrad
def Adagrad(w,b,dw,db, learning_rate,update_w,update_b,epsilon):
    update_w += dw**2
    update_b += db**2
    w = w - (learning_rate/np.sqrt(update_w + epsilon))*dw
    b = b - (learning_rate/np.sqrt(update_b + epsilon))*db
    return (w,b)