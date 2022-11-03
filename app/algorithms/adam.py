
import numpy as np
import math
#Adam
#Mini Batch Adam
def Adam(w,b,dw,db, learning_rate,update_w,update_b,epsilon,beta1,beta2,momentum_w,momentum_b,batch ):
    #Momentum
    momentum_w = beta1 * momentum_w + (1 - beta1) * dw
    momentum_b = beta1 * momentum_b + (1 - beta1) * db
    #Update History
    update_w = beta2 * update_w + (1 - beta2) * dw**2
    update_b = beta2 * update_b + (1 - beta2) * db**2 
    #Bias Correction
    momentum_w = momentum_w /(1 - math.pow(beta1,batch+1))  
    momentum_b = momentum_b /(1 - math.pow(beta1,batch+1))
    update_w = update_w /(1 - math.pow(beta2,batch+1))  
    update_b = update_b /(1 - math.pow(beta2,batch+1))
    #Update of Parameters
    w = w - (learning_rate/np.sqrt(update_w + epsilon))*momentum_w
    b = b - (learning_rate/np.sqrt(update_b + epsilon))*momentum_b
    return (w,b)