
import numpy as np
#Adagrad
#Mini Batch Nesterov accelerated gradient
def NAG(w,b,dw,db, learning_rate,w_look_ahead,b_look_ahead,prev_w_look_ahead,prev_b_look_ahead, mu):
    updated_w = mu*prev_w_look_ahead + learning_rate*dw #w_update_t = learning_rate*w_update_t-1 + eta*gradient(w_look_ahead)
    updated_b = mu  *prev_b_look_ahead + learning_rate*db #b_update_t = learning_rate*b_update_t-1 + eta*gradient(b_look_ahead)
    w = w - updated_w                                            #W_(t+1) = w_t - w_update_t
    b = b - updated_w                                            #B_(t+1) = b_t - b_update_t
    prev_w_look_ahead = updated_w
    prev_b_look_ahead = updated_b
    #New Look Ahead point after mini batch parameter update
    w_look_ahead = w - learning_rate*prev_w_look_ahead  
    b_look_ahead = b - learning_rate*prev_b_look_ahead
    return (w,b)


