import json
import streamlit as st
from utils.page import Page
from utils.logistic_regression import  main
from utils.sidebar import filter_table_option_adam, trainer
from utils.twoD_visuals import plot_Loss,plot_acc
from streamlit_lottie import st_lottie_spinner
# from utils.twoD_visuals import filter_table_option
# from utils.threeD_visuals import filter_table_option



class Adam(Page):
    def __init__(self, data, **kwargs):
        name = "Adam Gradient Descent"
        super().__init__(name, data, **kwargs)

    


    def content(self):

        content = "##### Gradient descent algorithm updates the parameters by moving in the direction opposite to the gradient of the objective function with respect to the network parameters."
        Page.content(st.markdown(content))
        col1, col2 = st.columns(2)
        with col1:
            Page.content(st.markdown("##### Parameter update rule will be given by,"))
            Page.content(st.write("For each parameter,"))
            Page.content(st.write("Updating the moemntum of both weight and bias"))   
            Page.content(st.latex(r"v_t = \beta_1 * v_{t-1} + (1-\beta_1) * (\nabla w_t)^2"))
            Page.content(st.write("Updating the history of both weight and bias"))  
            Page.content(st.latex(r"s_t = \beta_2 * v_{t-1} + (1-\beta_2) * (\nabla w_t)^2"))
            Page.content(st.write("Bias correction"))
            Page.content(st.latex(r"v_t = \frac{v_t}{1-\beta_1^t}"))
            Page.content(st.latex(r"s_t = \frac{s_t}{1-\beta_2^t}"))
            Page.content(st.latex(r"w_{t+1} = w_t - \frac{\eta}{\sqrt{s_t + \epsilon}}\nabla v_t "))
            
        with col2:
            Page.content(st.markdown("##### Gradient Descent Update Rule"))
            code = '''
import numpy as np
import math
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
    return (w,b)'''
            Page.content(st.code(code, language="python"))
        Page.content(st.markdown("##### In the batch gradient descent, we iterate over all the training data points and compute the cumulative sum of gradients for parameters ‘w’ and ‘b’. Then update the values of parameters based on the cumulative gradient value and the learning rate."))
        col3, col4 = st.columns(2)
        with col3:
            Page.content(st.markdown("### Pros:"))
            Page.content(st.markdown("##### ✽ Can handle sparse gradients on noisy datasets."))
            Page.content(st.markdown("##### ✽ Default hyperparameter values do well on most problems."))
            Page.content(st.markdown("##### ✽ Computationally efficient and requires little memory, thus memory efficient."))

        with col4:
            Page.content(st.markdown("### Cons:"))
            Page.content(st.markdown("##### ✽ Adam does not converge to an optimal solution in some areas (this is the motivation for AMSGrad) "))
            Page.content(st.markdown("##### ✽ Adam can suffer a weight decay problem (which is addressed in AdamW)."))
        hyperpara = filter_table_option_adam()
        train = trainer()
        if train:
            @st.experimental_memo
            def load_lottiefile(filepath: str):
                with open(filepath, "r") as f:
                    return json.load(f)
            lottie_json = load_lottiefile("app/data/simulation_animation.json")
            st.markdown("## 2- D graphs")
            with st_lottie_spinner(lottie_json, quality="high"):
                col1, col2 = st.columns(2)
                test_loss_list, test_accu_list,train_loss_list, train_accu_list,dw_list,db_list,batch_loss_list = main(hyperpara)
                with col1:
                    fig1 = plot_Loss(train_loss_list,test_loss_list,figsize=(650, 500))
                    Page.content(st.plotly_chart(fig1))
                with col2:
                    fig2 = plot_acc(train_accu_list,test_accu_list,figsize=(650, 500))
                    Page.content(st.plotly_chart(fig2))

