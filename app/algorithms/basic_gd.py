
# Vanilla Gradient Descent
def GD(w,b,dw,db, learning_rate):
    w = w - learning_rate * dw
    b = b - learning_rate*db
    return (w,b)