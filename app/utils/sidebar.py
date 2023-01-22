import streamlit as st


def sidebar_caption():
    """This is a demo of some shared sidebar elements.

    Reused this function to make sure we have the same sidebar elements if needed.
    """
    st.sidebar.header("Hyperparameters")
    st.sidebar.markdown("Please select the hyperparameter")


def filter_table_option():

    num_epoches = st.sidebar.slider('Number of Epochs', 1, 100, 1, key= 'num_epoches')
    batch_size = st.sidebar.slider('Batch size', 1, 100, 1, key= 'batch_size') 
    learning_rate = st.sidebar.slider('Learning rate', 1, 100, 1,key= 'learning_rate' )
    learning_decay = st.sidebar.slider('Learning rate decay factor', 1, 100, 1, key= 'learning_decay')
    decay_factor = st.sidebar.slider('Decay factor', 1, 100, 1, key= 'decay_factor')
    momentum = st.sidebar.slider('Momentum', 1, 100, 1, key= 'momentum')
    mu = st.sidebar.slider('Gamma', 1, 100, 1, key= 'mu')

    return num_epoches

def filter_table_option_vanilla():

    num_epoches = st.sidebar.number_input('Number of Epochs', 1, 100, step =1, value=15,key= 'num_epoches')
    batch_size = st.sidebar.number_input('Batch Size', 1, 1000, step =1,value=10, key= 'batch_size')
    learning_rate = st.sidebar.number_input('Learning rate', min_value = 0.5, max_value = 100.00, step = 0.01,value=2.5,key= 'learning_rate' )
    learning_decay_ = st.sidebar.checkbox("Learning decay")

    hyp = {'num_epoches': num_epoches,
            'learning_rate':learning_rate/1000,
            'learning_decay_':learning_decay_,
            'algo':"vanilla", 
            "batch_size":batch_size }
    if learning_decay_:
        decay_factor = st.sidebar.slider('Decay Factor', 0.00, 1.0, step =0.05,value=0.75,key= 'decay_factor' )
        hyp['decay_factor'] = decay_factor

    return hyp
def filter_table_option_mom():

    num_epoches = st.sidebar.number_input('Number of Epochs', 1, 100, step =1, value=15,key= 'num_epoches')
    batch_size = st.sidebar.number_input('Batch Size', 1, 1000, step =1,value=10, key= 'batch_size')
    learning_rate = st.sidebar.number_input('Learning rate', min_value = 0.5, max_value = 100.00, step = 0.01,value=2.5,key= 'learning_rate' )
    momentum = st.sidebar.slider('Momentum', 0.1, 1.0, step =0.1,value=0.9, key= 'momentum')
    learning_decay_ = st.sidebar.checkbox("Learning decay")

    hyp = {'num_epoches': num_epoches,
            'learning_rate':learning_rate/1000,
            'learning_decay_':learning_decay_,
            'mu':momentum,
            'algo':"momentum", 
            "batch_size":batch_size }
    if learning_decay_:
        decay_factor = st.sidebar.slider('Decay Factor', 0.00, 1.0, step =0.05,value=0.75,key= 'decay_factor' )
        hyp['decay_factor'] = decay_factor

    return hyp

def filter_table_option_nag():

    num_epoches = st.sidebar.number_input('Number of Epochs', 1, 100, step =1, value=15,key= 'num_epoches')
    batch_size = st.sidebar.number_input('Batch Size', 1, 1000, step =1,value=10, key= 'batch_size')
    learning_rate = st.sidebar.number_input('Learning rate', min_value = 0.5, max_value = 100.00, step = 0.01,value=2.5,key= 'learning_rate' )
    momentum = st.sidebar.slider('Momentum', 0.1, 1.0, step =0.1,value=0.9, key= 'momentum')
    learning_decay_ = st.sidebar.checkbox("Learning decay")

    hyp = {'num_epoches': num_epoches,
            'learning_rate':learning_rate/1000,
            'learning_decay_':learning_decay_,
            'mu':momentum,
            'algo':"nag", 
            "batch_size":batch_size }
    if learning_decay_:
        decay_factor = st.sidebar.slider('Decay Factor', 0.00, 1.0, step =0.05,value=0.75,key= 'decay_factor' )
        hyp['decay_factor'] = decay_factor

    return hyp

def filter_table_option_adagrad():

    num_epoches = st.sidebar.number_input('Number of Epochs', 1, 100, step =1, value=15,key= 'num_epoches')
    batch_size = st.sidebar.number_input('Batch Size', 1, 1000, step =1,value=10, key= 'batch_size')
    learning_rate = st.sidebar.number_input('Learning rate', min_value = 0.5, max_value = 100.00, step = 0.01,value=2.5,key= 'learning_rate' )
    epsilon = st.sidebar.slider('Epsilon', 0.01, 1.0, step =0.1,value=0.9, key= 'epsilon')
    learning_decay_ = st.sidebar.checkbox("Learning decay")

    hyp = {'num_epoches': num_epoches,
            'learning_rate':learning_rate/1000,
            'learning_decay_':learning_decay_,
            'epsilon':epsilon,
            'algo':"adagrad", 
            "batch_size":batch_size }
    if learning_decay_:
        decay_factor = st.sidebar.slider('Decay Factor', 0.00, 1.0, step =0.05,value=0.75,key= 'decay_factor' )
        hyp['decay_factor'] = decay_factor

    return hyp

def filter_table_option_rmsprop():

    num_epoches = st.sidebar.number_input('Number of Epochs', 1, 100, step =1, value=15,key= 'num_epoches')
    batch_size = st.sidebar.number_input('Batch Size', 1, 1000, step =1,value=10, key= 'batch_size')
    learning_rate = st.sidebar.number_input('Learning rate', min_value = 0.5, max_value = 100.00, step = 0.01,value=2.5,key= 'learning_rate' )
    epsilon = st.sidebar.slider('Epsilon', 0.01, 1.0, step =0.1,value=0.9, key= 'epsilon')
    beta = st.sidebar.slider('Beta', 0.01, 1.0, step =0.1,value=0.9, key= 'beta')
    learning_decay_ = st.sidebar.checkbox("Learning decay")

    hyp = {'num_epoches': num_epoches,
            'learning_rate':learning_rate/1000,
            'learning_decay_':learning_decay_,
            'epsilon':epsilon,
            'beta':beta,
            'algo':"rmsprop", 
            "batch_size":batch_size }
    if learning_decay_:
        decay_factor = st.sidebar.slider('Decay Factor', 0.00, 1.0, step =0.05,value=0.75,key= 'decay_factor' )
        hyp['decay_factor'] = decay_factor

    return hyp

def filter_table_option_adam():

    num_epoches = st.sidebar.number_input('Number of Epochs', 1, 100, step =1, value=15,key= 'num_epoches')
    batch_size = st.sidebar.number_input('Batch Size', 1, 1000, step =1,value=10, key= 'batch_size')
    learning_rate = st.sidebar.number_input('Learning rate', min_value = 0.5, max_value = 100.00, step = 0.01,value=2.5,key= 'learning_rate' )
    epsilon = st.sidebar.slider('Epsilon', 0.01, 1.0, step =0.1,value=0.9, key= 'epsilon')
    beta1 = st.sidebar.slider('Beta1', 0.01, 1.0, step =0.1,value=0.9, key= 'beta')
    beta2 = st.sidebar.slider('Beta2', 0.01, 1.0, step =0.1,value=0.9, key= 'beta2')
    learning_decay_ = st.sidebar.checkbox("Learning decay")

    hyp = {'num_epoches': num_epoches,
            'learning_rate':learning_rate/1000,
            'learning_decay_':learning_decay_,
            'epsilon':epsilon/10000000,
            'beta1':beta1,
            'beta2':beta2,
            'algo':"adam", 
            "batch_size":batch_size }
    if learning_decay_:
        decay_factor = st.sidebar.slider('Decay Factor', 0.00, 1.0, step =0.05,value=0.75,key= 'decay_factor' )
        hyp['decay_factor'] = decay_factor

    return hyp


def trainer():
    return st.sidebar.button('🚀   Train the MNIST model')



