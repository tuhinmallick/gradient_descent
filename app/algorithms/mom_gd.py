
#Momentum_GD
#Mini Batch Momentum Based Gradient Discent
def Momentum_GD(w,b,dw,db, learning_rate,mu,w_velocity,b_velocity):
    w_velocity = mu * w_velocity + learning_rate * dw
    b_velocity = mu * b_velocity + learning_rate * db
    w -= w_velocity
    b-= b_velocity
    return (w,b)

    