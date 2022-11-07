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
            Page.content(st.write("With  Momemntum,"))
            Page.content(st.latex(r"update_{t +1} = \gamma * update_t + \eta\nabla w_{t}"))
            Page.content(st.latex(r"w_{t +1} = w_t - update_{t +1}"))
            Page.content(st.write("With  NAG,"))
            Page.content(st.latex(r"w_{lookahead} =  w_t -\gamma * update_{t-1}"))
            Page.content(st.latex(r"update_t =\gamma * update_{t-1} + \eta\nabla w_{lookahead}"))
            Page.content(st.latex(r"w_{t+1} =w_t -update_t"))

            
        with col2:
            Page.content(st.markdown("##### Gradient Descent Update Rule"))
            code = '''

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

'''
            Page.content(st.code(code, language="python"))
        Page.content(st.markdown("##### Instead of evaluating gradient at the current position (red circle), we know that our momentum is about to carry us to the tip of the green arrow. With Nesterov momentum we therefore instead evaluate the gradient at this looked-ahead position. "))
        Page.content(st.markdown("##### Again, we set the momentum term γ to a value of around 0.9. While Momentum first computes the current gradient (small blue vector in Image 4) and then takes a big jump in the direction of the updated accumulated gradient (big blue vector), NAG first makes a big jump in the direction of the previous accumulated gradient (brown vector), measures the gradient and then makes a correction (red vector), which results in the complete NAG update (green vector). This anticipatory update prevents us from going too fast and results in increased responsiveness, which has significantly increased the performance of RNNs on a number of tasks"))
        Page.content(st.image("app/data/nesterov.png", use_column_width="always"))
        col3, col4 = st.columns(2)
        with col3:
            Page.content(st.markdown("### Pros:"))
            Page.content(st.markdown("##### ✽ Has the notion of where it is going so that it knows when to slow down before the hill slopes up again"))
        with col4:
            Page.content(st.markdown("### Cons:"))
            Page.content(st.markdown("##### ✽ Does not update each individual parameter to perform larger or smaller updates depending on their importance."))
            Page.content(st.markdown("##### ✽ Is not adaptive"))
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
                test_loss_list, test_accu_list,train_loss_list, train_accu_list,dw_list,db_list,batch_loss_list = main(hyperpara)
                with col1:
                    fig1 = plot_Loss(train_loss_list,test_loss_list,figsize=(650, 500))
                    Page.content(st.plotly_chart(fig1))
                with col2:
                    fig2 = plot_acc(train_accu_list,test_accu_list,figsize=(650, 500))
                    Page.content(st.plotly_chart(fig2))

