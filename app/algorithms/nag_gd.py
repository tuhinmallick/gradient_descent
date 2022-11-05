
#Adagrad
#Mini Batch Nesterov accelerated gradient
def NAG(w,b,dw,db, learning_rate,w_look_ahead,b_look_ahead,prev_w_look_ahead,prev_b_look_ahead, mu):
    #New Look Ahead point after mini batch parameter update
    w_look_ahead = w - mu*prev_w_look_ahead  
    b_look_ahead = b - mu*prev_b_look_ahead
    updated_w = learning_rate*prev_w_look_ahead + mu*w_look_ahead #w_update_t = learning_rate*w_update_t-1 + eta*gradient(w_look_ahead)
    updated_b = learning_rate  *prev_b_look_ahead + mu*b_look_ahead #b_update_t = learning_rate*b_update_t-1 + eta*gradient(b_look_ahead)
    w = w - updated_w                                            #W_(t+1) = w_t - w_update_t
    b = b - updated_w                                            #B_(t+1) = b_t - b_update_t
    prev_w_look_ahead = updated_w
    prev_b_look_ahead = updated_b
   
    return (w,b)


