import matplotlib.pyplot as plt
import plotly.graph_objects as go
def plot_Loss(train_loss_list,test_loss_list, figsize=(1400, 500)):
    """store the plots"""
    epoch_list = list(range(len(train_loss_list)))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epoch_list,
            y=train_loss_list,
            name="Training loss",
            marker_color="rgba(242,242,242,1)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epoch_list,
            y=test_loss_list,
            name="Test loss",
            marker_color="rgba(141,201,40,0.9)",
        )
    )
    fig.update_layout(
        autosize=False,
        width=figsize[0],
        height=figsize[1],
        xaxis_title="Epoch",
        yaxis_title="Loss Function",
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", xanchor="right", y=1.0, x=1
        ),
        template="plotly_dark",
        margin=dict(l=80, r=30, t=50, b=50),
        plot_bgcolor="#151934",
        paper_bgcolor="#151934",
        title={
            'text': 'Loss Function ~ Epoch',
            'y': 0.95,
            'x': 0.5,
            "xanchor": 'center',
            'yanchor': 'top',
        },
    )
    return fig

def plot_acc(train_accu_list,test_accu_list, figsize=(1400, 500)):
    """store the plots"""
    epoch_list = list(range(len(train_accu_list)))
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=epoch_list,
            y=train_accu_list,
            name="Training accuracy",
            marker_color="rgba(242,242,242,1)",
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=epoch_list,
            y=test_accu_list,
            name="Test accuracy",
            marker_color="rgba(141,201,40,0.9)",
        )
    )
    fig1.update_layout(
        autosize=False,
        width=figsize[0],
        height=figsize[1],
        xaxis_title="Epoch",
        yaxis_title="Test Accuracy",
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", xanchor="right", y=1.0, x=1
        ),
        template="plotly_dark",
        margin=dict(l=80, r=30, t=50, b=50),
        plot_bgcolor="#151934",
        paper_bgcolor="#151934",
        title={
            'text': 'Test Accuracy ~ Epoch',
            'y': 0.95,
            'x': 0.3,
            "xanchor": 'center',
            'yanchor': 'top',
        },
    )
    return fig1
def plot_params(dw_list,db_list,batch_loss_list,epoch, figsize=(1400, 500)):
    """store the plots"""
    import pdb
    pdb.set_trace()
    epoch_list = epoch
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=epoch_list,
            y=batch_loss_list,
            name="Error",
            marker_color="rgba(141,201,40,0.0)",
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=epoch_list,
            y=dw_list,
            name="Weight",
            marker_color="rgba(141,201,40,0.9)",
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=epoch_list,
            y=db_list,
            name="Bias",
            marker_color="rgba(42,242,242,1)",
        )
    )
    fig1.update_layout(
        autosize=False,
        width=figsize[0],
        height=figsize[1],
        xaxis_title="Epoch",
        yaxis_title="Test Accuracy",
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", xanchor="right", y=1.0, x=1
        ),
        template="plotly_dark",
        margin=dict(l=80, r=30, t=50, b=50),
        plot_bgcolor="#151934",
        paper_bgcolor="#151934",
        title={
            'text': 'Test Accuracy ~ Epoch',
            'y': 0.95,
            'x': 0.5,
            "xanchor": 'center',
            'yanchor': 'top',
        },
    )
    return fig1