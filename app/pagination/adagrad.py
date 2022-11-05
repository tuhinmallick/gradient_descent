import json
import streamlit as st
from utils.page import Page
from utils.logistic_regression import  main
from utils.sidebar import filter_table_option_adagrad, trainer
from utils.twoD_visuals import plot_Loss,plot_acc
from streamlit_lottie import st_lottie_spinner
# from utils.twoD_visuals import filter_table_option
# from utils.threeD_visuals import filter_table_option



class Adagrad(Page):
    def __init__(self, data, **kwargs):
        name = "Adagrad Gradient Descent"
        super().__init__(name, data, **kwargs)

    


    def content(self):

        content = "##### Adagrad is an algorithm for gradient-based optimization that does just this: It adapts the learning rate to the parameters, performing smaller updates (i.e. low learning rates) for parameters associated with frequently occurring features, and larger updates (i.e. high learning rates) for parameters associated with infrequent features."
        Page.content(st.markdown(content))
        col1, col2 = st.columns(2)
        with col1:
            Page.content(st.markdown("##### Parameter update rule will be given by,"))
            Page.content(st.latex(r"G_t = G_{t-1} + (\nabla w_t)^2"))
            Page.content(st.latex(r"w_{new} = w_{old} - \frac{\eta}{\sqrt{G_t +\epsilon}} \nabla w_t"))
            
        with col2:
            Page.content(st.markdown("##### Gradient Descent Update Rule"))
            code = '''
import numpy as np
#Mini Batch Adagrad
def Adagrad(w,b,dw,db, learning_rate,update_w,update_b,epsilon):
    update_w += dw**2
    update_b += db**2
    w = w - (learning_rate/np.sqrt(update_w + epsilon))*dw
    b = b - (learning_rate/np.sqrt(update_b + epsilon))*db
    return (w,b)'''
            Page.content(st.code(code, language="python"))
        Page.content(st.markdown("##### In Adagrad, we are maintaining the running squared sum of gradients and then we update the parameters by dividing the learning rate with the square root of the historical values. Instead of having a static learning rate here we have dynamic learning for dense and sparse features."))
        col3, col4 = st.columns(2)
        with col3:
            Page.content(st.markdown("### Pros:"))
            Page.content(st.markdown("##### ✽ Updates each individual parameter to perform larger or smaller updates on its importance"))
            Page.content(st.markdown("##### ✽ Is adaptive"))
        with col4:
            Page.content(st.markdown("### Cons:"))
            Page.content(st.markdown("##### ✽ Accumulationof squared gradients: Since every added term is positive, the accumulated sum keeps growing."))
            Page.content(st.markdown("##### ✽ Leads to skrinkage of learning rate leading to its value being infinitesimally small."))
        hyperpara = filter_table_option_adagrad()
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

