import json
import streamlit as st
from utils.page import Page
from utils.logistic_regression import  main
from utils.sidebar import filter_table_option_mom, trainer
from utils.twoD_visuals import plot_Loss,plot_acc
from streamlit_lottie import st_lottie_spinner
# from utils.twoD_visuals import filter_table_option
# from utils.threeD_visuals import filter_table_option



class Momentum(Page):
    def __init__(self, data, **kwargs):
        name = "Momentum based Gradient Descent"
        super().__init__(name, data, **kwargs)

    


    def content(self):

        content = "##### In Momentum GD, we are moving with an exponential decaying cumulative average of previous gradients and current gradient."
        Page.content(st.markdown(content))
        col1, col2 = st.columns(2)
        with col1:
            Page.content(st.markdown("##### Parameter update rule will be given by,"))
            Page.content(st.image("app/data/mom.png", use_column_width="always"))
            
        with col2:
            Page.content(st.markdown("##### Gradient Descent Update Rule"))
            Page.content(st.image("app/data/mom_code.png", use_column_width="always"))
        Page.content(st.markdown("##### In the batch gradient descent, we iterate over all the training data points and compute the cumulative sum of gradients for parameters ‘w’ and ‘b’. Then update the values of parameters based on the cumulative gradient value and the learning rate."))
        col3, col4 = st.columns(2)
        with col3:
            Page.content(st.markdown("### Pros:"))
            Page.content(st.markdown("##### ✽ Helps to accelerate in the relevant direction and dampens oscillation."))
            Page.content(st.markdown("##### ✽ Faster convergence"))
        with col4:
            Page.content(st.markdown("### Cons:"))
            Page.content(st.markdown("##### ✽ Does not have the notion of where it is going so does not slows down when hipp slopes up again."))
            Page.content(st.markdown("##### ✽ Does not update each individual parameter to perform larger or smaller updates depending on their importance."))
            Page.content(st.markdown("##### ✽ Is not adaptive"))
        hyperpara = filter_table_option_mom()
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
                    fig1 = plot_Loss(loss_list,figsize=(650, 500))
                    Page.content(st.plotly_chart(fig1))
                with col2:
                    fig2 = plot_acc(accu_list,figsize=(650, 500))
                    Page.content(st.plotly_chart(fig2))

