
import numpy as np
#RMSprop
#Mini Batch RMSprop
def RMSprop(w,b,dw,db, learning_rate,update_w,update_b,epsilon,beta):
    update_w = beta * update_w + (1 - beta)* dw**2
    update_b = beta * update_b + (1- beta) * db**2
    w = w - (learning_rate/np.sqrt(update_w + epsilon))*dw
    b = b - (learning_rate/np.sqrt(update_b + epsilon))*db
    return (w,b)

    