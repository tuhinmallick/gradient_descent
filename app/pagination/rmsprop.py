import json
import streamlit as st
from utils.page import Page
from utils.logistic_regression import  main
from utils.sidebar import filter_table_option_rmsprop, trainer
from utils.twoD_visuals import plot_Loss,plot_acc
from streamlit_lottie import st_lottie_spinner
# from utils.twoD_visuals import filter_table_option
# from utils.threeD_visuals import filter_table_option



class RMSprop(Page):
    def __init__(self, data, **kwargs):
        name = "RMSprop Gradient Descent"
        super().__init__(name, data, **kwargs)

    


    def content(self):

        content = "##### Gradient descent algorithm updates the parameters by moving in the direction opposite to the gradient of the objective function with respect to the network parameters."
        Page.content(st.markdown(content))
        col1, col2 = st.columns(2)
        with col1:
            Page.content(st.markdown("##### Parameter update rule will be given by,"))
            Page.content(st.write("For each parameter,"))
            Page.content(st.latex(r"v_t = \beta * v_{t-1} + (1-\beta) * (\nabla w_t)^2"))
            Page.content(st.latex(r"w_{t+1} = w_t - \frac{\eta}{\sqrt{v_t + \epsilon}}\nabla w_t "))
            
        with col2:
            Page.content(st.markdown("##### Gradient Descent Update Rule"))
            code = '''
import numpy as np
#RMSprop
#Mini Batch RMSprop
def RMSprop(w,b,dw,db, learning_rate,update_w,update_b,epsilon,beta):
    update_w = beta * update_w + (1 - beta)* dw**2
    update_b = beta * update_b + (1- beta) * db**2
    w = w - (learning_rate/np.sqrt(update_w + epsilon))*dw
    b = b - (learning_rate/np.sqrt(update_b + epsilon))*db
    return (w,b)

    '''
            Page.content(st.code(code, language="python"))
        Page.content(st.markdown("##### The only change we need to do in AdaGrad code is how we update the variables   v_w   and   v_b  . In AdaGrad   v_w   and v_b is always increasing by squares of the gradient per parameter wise since the first epoch but in RMSProp   v_w   and   v_b   is exponentially decaying weighted sum of gradients by using a hyperparameter called ‘gamma’."))
        col3, col4 = st.columns(2)
        with col3:
            Page.content(st.markdown("### Pros:"))
            Page.content(st.markdown("##### ✽ Instead of storing inefficiently all previous gradients - recursively defines as a decaying average of all past squared gradients - learning rate optimally high."))
        with col4:
            Page.content(st.markdown("### Cons:"))
            Page.content(st.markdown("##### ✽ Does not keep an exponentiallydecaying average of past gradients"))
        hyperpara = filter_table_option_rmsprop()
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