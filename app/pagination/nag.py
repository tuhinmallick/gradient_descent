import json
import streamlit as st
from utils.page import Page
from utils.logistic_regression import  main
from utils.sidebar import  filter_table_option_nag, trainer
from utils.twoD_visuals import plot_Loss,plot_acc
from streamlit_lottie import st_lottie_spinner
# from utils.twoD_visuals import filter_table_option
# from utils.threeD_visuals import filter_table_option



class NAG(Page):
    def __init__(self, data, **kwargs):
        name = "Nesterov Accelerated Gradient Descent"
        super().__init__(name, data, **kwargs)

    


    def content(self):

        content = "##### In Nesterov Accelerated Gradient Descent we are looking forward to seeing whether we are close to the minima or not before we take another step based on the current gradient value so that we can avoid the problem of overshooting."
        Page.content(st.markdown(content))
        col1, col2 = st.columns(2)
        with col1:
            Page.content(st.markdown("##### Parameter update rule will be given by,"))
            Page.content(st.image("app/data/nag.png", use_column_width="always"))
            
        with col2:
            Page.content(st.markdown("##### Gradient Descent Update Rule"))
            Page.content(st.image("app/data/nag_code.png", use_column_width="always"))
        Page.content(st.markdown("##### Instead of evaluating gradient at the current position (red circle), we know that our momentum is about to carry us to the tip of the green arrow. With Nesterov momentum we therefore instead evaluate the gradient at this looked-ahead position. "))
        Page.content(st.image("app/data/nesterov.jpeg", use_column_width="always"))
        col3, col4 = st.columns(2)
        with col3:
            Page.content(st.markdown("### Pros:"))
            Page.content(st.markdown("##### Guaranteed to converge to global minimum for convex error surfaces and to a local minimum for non-convex surfaces."))
        with col4:
            Page.content(st.markdown("### Cons:"))
            Page.content(st.markdown("##### Very slow."))
        hyperpara = filter_table_option_nag()
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
                loss_list,accu_list = main(hyperpara)
                with col1:
                    fig1 = plot_Loss(loss_list,figsize=(650, 500))
                    Page.content(st.plotly_chart(fig1))
                with col2:
                    fig2 = plot_acc(accu_list,figsize=(650, 500))
                    Page.content(st.plotly_chart(fig2))

