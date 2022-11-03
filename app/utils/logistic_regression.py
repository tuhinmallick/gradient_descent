import numpy as np
import h5py
# data file type h5py
import matplotlib.pyplot as plt
from algorithms import adagrad,adam,basic_gd,nag_gd,rmsprop, mom_gd
import streamlit as st


def load_mnist(filename):
    """load MNIST data"""
    MNIST_data = h5py.File(filename, 'r')
    x_train = np.float32(MNIST_data['x_train'][:])
    y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
    x_test = np.float32(MNIST_data['x_test'][:])
    y_test = np.int32(np.array(MNIST_data['y_test'][:,0]))
    MNIST_data.close()
    return x_train,y_train,x_test,y_test

def initialize(num_inputs,num_classes):
    """initialize the parameters"""
    # num_inputs = 28*28 = 784
    # num_classes = 10
    w = np.random.randn(num_classes, num_inputs) / np.sqrt(num_classes*num_inputs) # (10*784)
    b = np.random.randn(num_classes, 1) / np.sqrt(num_classes) # (10*1) 
    
    param = {
        'w' : w, # (10*784)
        'b' : b  # (10*1)
    }
    return param

def softmax(z):
    """implement the softmax functions
    input: numpy ndarray
    output: numpy ndarray
    """
    exp_list = np.exp(z)
    result = 1/sum(exp_list) * exp_list
    # import pdb;pdb.set_trace()
    result = result.reshape((len(z),1))
    assert (result.shape == (len(z),1))
    return result

def neg_log_loss(pred, label):
    """implement the negative log loss"""
    loss = -np.log(pred[int(label)])
    return loss

def mini_batch_gradient(param, x_batch, y_batch):
    """implement the function to compute the mini batch gradient
    input: param -- parameters dictionary (w, b)
           x_batch -- a batch of x (size, 784)
           y_batch -- a batch of y (size,)
    output: dw, db, batch_loss
    """
    batch_size = x_batch.shape[0]
    w_grad_list = []
    b_grad_list = []
    batch_loss = 0
    for i in range(batch_size):
        x,y = x_batch[i],y_batch[i]
        x = x.reshape((784,1)) # x: (784,1)
        E = np.zeros((10,1)) #(10*1)
        E[y][0] = 1 
        # import pdb;pdb.set_trace()
        pred = softmax(np.matmul(param['w'], x)+param['b']) #(10*1)

        loss = neg_log_loss(pred, y)
        batch_loss += loss

        w_grad = E - pred
        w_grad = - np.matmul(w_grad, x.reshape((1,784)))
        w_grad_list.append(w_grad)

        b_grad = -(E - pred)
        b_grad_list.append(b_grad)

    dw = sum(w_grad_list)/batch_size
    db = sum(b_grad_list)/batch_size
    return dw, db, batch_loss

def eval(param, x_data, y_data):
    """ implement the evaluation function
    input: param -- parameters dictionary (w, b)
           x_data -- x_train or x_test (size, 784)
           y_data -- y_train or y_test (size,)
    output: loss and accuracy
    """
    # w: (10*784), x: (10000*784), y:(10000,)
    loss_list = []
    w = param['w'].transpose()
    dist = np.array([np.squeeze(softmax(np.matmul(x_data[i], w))) for i in range(len(y_data))])
    result = np.argmax(dist,axis=1)
    accuracy = sum(result == y_data)/float(len(y_data))

    loss_list = [neg_log_loss(dist[i],y_data[i]) for i in range(len(y_data))]
    loss = sum(loss_list)
    return loss, accuracy

def train(param, hyp , x_train, y_train, x_test, y_test):
    """ implement the train function
    input: param -- parameters dictionary (w, b)
           hyp -- hyperparameters dictionary
           x_train -- (60000, 784)
           y_train -- (60000,)
           x_test -- x_test (10000, 784)
           y_test -- y_test (10000,)
    output: test_loss_list, test_accu_list
    """
    num_epoches = hyp['num_epoches']
    batch_size = hyp['batch_size']
    learning_rate = hyp['learning_rate']
    dw_list, db_list, batch_loss_list = [],[],[]
    test_loss_list, test_accu_list = [],[]
    train_loss_list, train_accu_list = [],[]
    print(" Gradien Descent Type: ")
    if bool(hyp['algo']) == "momentum":
        mu = hyp['mu']
        w_velocity = np.zeros(param['w'].shape)
        b_velocity = np.zeros(param['b'].shape) 
    if bool(hyp['algo']) == "adagrad":
        epsilon = 0.5
        update_w, update_b = 0,0
    if bool(hyp['algo']) == "adam":
        print(" ADAM")
        epsilon = 1e-8
        beta1 = 0.9
        beta2 = 0.999
        momentum_w,momentum_b = 0,0
        update_w, update_b = 0,0
    if bool(hyp['algo']) == "nag":
        mu = hyp['mu']
        prev_w_look_ahead,prev_b_look_ahead = 0,0
        w_look_ahead = param['w'] - learning_rate*prev_w_look_ahead                      #W_look_ahead = w_t - lr*w_update_t-1
        b_look_ahead = param['b'] - learning_rate*prev_b_look_ahead                      #B_look_ahead = b_t - lr*b_update_t-1
    if bool(hyp['algo']) == "rmsprop":
        epsilon = 0.5
        beta = 0.95
        update_w, update_b = 0,0

    for epoch in range(num_epoches):
        
        # select the random sequence of training set
        rand_indices = np.random.choice(x_train.shape[0],x_train.shape[0],replace=False)
        num_batch = int(x_train.shape[0]/batch_size)
        batch_loss100 = 0
        
        if bool(hyp['learning_decay_']) == True:
            try:
                if test_accu_list[-1] - test_accu_list[-2] < 0.001:
                    learning_rate *= hyp['decay_factor']
            except:
                pass
            
            message = 'learning rate: %.8f' % learning_rate
            print(message)

        # for each batch of train data
        for batch in range(num_batch):
            index = rand_indices[batch_size*batch:batch_size*(batch+1)]
            x_batch = x_train[index]
            y_batch = y_train[index]

            # calculate the gradient w.r.t w and b
            dw, db, batch_loss = mini_batch_gradient(param, x_batch, y_batch)
            batch_loss100 += batch_loss
            # update the parameters with the learning rate
            if bool(hyp['algo']) == "momentum":
                param['w'], param['b'] = mom_gd.Momentum_GD(param['w'],param['b'],dw,db,learning_rate,mu, w_velocity,b_velocity)
            elif bool(hyp['algo']) == "adagrad":
                param['w'], param['b'] = adagrad.Adagrad(param['w'],param['b'],dw,db,learning_rate,update_w,update_b,  epsilon )
            elif bool(hyp['algo']) == "adam":
                param['w'], param['b'] = adam.Adam(param['w'],param['b'],dw,db,learning_rate,update_w,update_b,epsilon,beta1,beta2,momentum_w,momentum_b,batch)
            elif bool(hyp['algo']) == "nag":
                param['w'], param['b'] = nag_gd.NAG(param['w'],param['b'],dw,db,learning_rate,w_look_ahead,b_look_ahead,prev_w_look_ahead,prev_b_look_ahead, mu)
            elif bool(hyp['algo']) == "rmsprop":
                param['w'], param['b'] = rmsprop.RMSprop(param['w'],param['b'],dw,db,learning_rate,update_w,update_b,epsilon,beta)
            else:
                param['w'], param['b'] = basic_gd.GD(param['w'],param['b'],dw=dw,db=db,learning_rate=learning_rate)
            if batch % 100 == 0:
                message = 'Epoch %d, Batch %d, Loss %.2f' % (epoch+1, batch, batch_loss)
                print(message)
                dw_list.append(dw) 
                db_list.append(db)
                batch_loss_list.append(batch_loss)
                batch_loss100 = 0
        train_loss, train_accu = eval(param,x_train,y_train)
        test_loss, test_accu = eval(param,x_test,y_test)
        test_loss_list.append(test_loss)
        test_accu_list.append(test_accu)
        train_loss_list.append(train_loss/10)
        train_accu_list.append(train_accu)

        message = 'Epoch %d, Train Loss %.2f, Train Accu %.4f, Test Loss %.2f, Test Accu %.4f' % (epoch+1, train_loss, train_accu, test_loss, test_accu)
        print(message)
    return test_loss_list, test_accu_list,train_loss_list, train_accu_list,dw_list,db_list,batch_loss_list

def main(hyperpara): 
    # loading MNIST data
    x_train,y_train,x_test,y_test = load_mnist('app/data/MNISTdata.hdf5')

    # setting the random seed
    np.random.seed(1024)

    # initialize the parameters
    num_inputs = x_train.shape[1]
    num_classes = len(set(y_train))
    param = initialize(num_inputs,num_classes)

    # train the model
    return(train(param,hyperpara,x_train,y_train,x_test,y_test))

    # plot the loss and accuracy

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, 
                        default="sample", help="Config of hyperparameters")
    args = parser.parse_args()

    
    
    main(args)