
from math import ceil
import logomaker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'


def plot_model_perf_with_depth(config):
    models_df = pd.read_csv(
        f'{config.output_folder}/modelPerformance.csv'
    )
    train_r2_trace = go.Scatter(
        x=models_df['identifier'],
        y=models_df['trainr2'],
        line={'color': 'black', 'dash': 'dash'},
        marker={'color': 'black'},
        mode='lines+markers',
        name='Train R2',
    )
    test_r2_trace = go.Scatter(
        x=models_df['identifier'],
        y=models_df['testr2'],
        line={'color': 'black'},
        marker={'color': 'black'},
        mode='lines+markers',
        name='Test R2',
    )
    train_pearson_trace = go.Scatter(
        x=models_df['identifier'],
        y=models_df['trainpearson'],
        line={'color': 'blue', 'dash': 'dash'},
        marker={'color': 'blue'},
        mode='lines+markers',
        name='Train Pearson',
    )
    test_pearson_trace = go.Scatter(
        x=models_df['identifier'],
        y=models_df['testpearson'],
        line={'color': 'blue'},
        marker={'color': 'blue'},
        mode='lines+markers',
        name='Test Pearson',
    )
    model_size_trace = go.Scatter(
        x=models_df['identifier'],
        y=models_df['modelSize']/1_048_576,
        line={'color': 'darkgreen'},
        marker={'color': 'darkgreen'},
        mode='lines+markers',
        name='Model Size',
    )
    return [train_r2_trace, test_r2_trace, train_pearson_trace, test_pearson_trace], [model_size_trace]

def create_logo_plot(config):
    fig, plt_axes = plt.subplots(1, 4, figsize=(12, 3))

    train_dfs = []
    for i in range(1, 6):
        x = pd.read_csv(f'{config.output_folder}/flippedSeqs.csv', usecols=['peptide'])
        x['peptide'] = x['peptide'].apply(
            lambda pep : pep.replace('.', '').replace('_', '').replace('[UNIMOD:35]', '').replace('[UNIMOD:4]', '').replace('m', 'M')
        )
        x['len'] = x['peptide'].apply(len)
        train_dfs.append(x)
    full_train_df = pd.concat(train_dfs)
    y_lim = 0.0
    for len_idx, pep_len in enumerate(range(8, 12)):
        len_df = full_train_df[full_train_df['len'] == pep_len]
        total = len_df.shape[0]
        if total == 0:
            continue

        aa_counts = np.zeros([pep_len, len(AMINO_ACIDS)])
        for _, df_row in len_df.iterrows():
            for pos_idx, aa in enumerate(df_row['peptide']):
                aa_counts[pos_idx, AMINO_ACIDS.index(aa)] += 1

        count_df = pd.DataFrame(
            aa_counts,
            columns=list(AMINO_ACIDS)
        )

        count_df.index += 1
        # print(ww_df)
        info_df = logomaker.transform_matrix(
            count_df,
            from_type='counts',
            to_type='information',
        )

        logo_plot = logomaker.Logo(
            info_df,
            ax=plt_axes[len_idx],
            font_name='DejaVu Sans',
            color_scheme='dmslogo_funcgroup',
            vpad=.1,
            width=.8
        )

        logo_plot.style_xticks(anchor=1, spacing=1)
        # ww_logo.highlight_position(p=4, color='gold', alpha=.5)
        # ww_logo.highlight_position(p=26, color='gold', alpha=.5)

        # style using Axes methods
        logo_plot.ax.set_xlim([0, len(info_df)+1])

        y_lim = max(y_lim, ceil(info_df.sum(axis=1).max()*5)/5)
        
        # Hide the right and top spines
        logo_plot.ax.spines.right.set_visible(False)
        logo_plot.ax.spines.top.set_visible(False)

        # Only show ticks on the left and bottom spines
        logo_plot.ax.yaxis.set_ticks_position('left')
        logo_plot.ax.xaxis.set_ticks_position('bottom')

    for ax in plt_axes:
        ax.set_ylim([0, y_lim])
    # plt.tight_layout()
    # plt.savefig(f'{config.output_folder}/logo.eps', format='eps')
    plt.show()

def get_imp_trace(config):
    imp_df = pd.read_csv(
        f'{config.output_folder}/importances/model{config.best_model}.csv',
    )
    return go.Bar(
        x=imp_df['importances'],
        y=imp_df['feature'],
        orientation='h',
        marker_color='black',
        name='Feature Importance'
    )

def plot_best_performance(config):
    preds_df = pd.read_csv(f'{config.output_folder}/testPreds.csv')
    preds_df['locint'] = preds_df['bIntesAtLoc'] + preds_df['yIntesAtLoc']

    return [
        go.Scatter(
            x=preds_df[f'predictedDiff{config.best_model}'],
            y=preds_df['specAngleDiff'],
            mode = 'markers',
            marker_color=preds_df['locint'],
            marker_showscale=True,
            marker_colorscale='Bluered',
            marker_cmin=0,
            marker_cmax=1,
            marker_colorbar={'x':0.4, 'y':0.5, 'len': 0.25, 'ticks': 'outside'},
            name='Test Data',
        ),
        go.Scatter(
            x=preds_df[f'predictedDiff{config.best_model}'],
            y=preds_df['specAngleDiff'],
            mode = 'markers',
            marker_color=preds_df['spectralAngle'],
            marker_showscale=True,
            marker_colorscale='Bluered',
            marker_cmin=0,
            marker_cmax=1,
            marker_colorbar={'y':0.5, 'len': 0.25, 'ticks': 'outside'},
            name='Test Data',
        ),
    ]


def analyse(config):
    # create_logo_plot(config)

    perf_trace, size_trace = plot_model_perf_with_depth(config)

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[
            'Model Performance with Depth',
            'Model Size with Depth',
            'Coloured by Intensity at Location',
            'Coloured by Spectral Angle',
            'Feature Importances'
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.2,
    )

    for trace in perf_trace:
        fig.add_trace(
            trace,
            row=1,
            col=1,
        )
    if config.best_model is not None:
        fig.add_vline(
            x=config.best_model,
            line_color='red',
            line_dash='dash',
            row=1,
            col=1,
        )
    for trace in size_trace:
        fig.add_trace(
            trace,
            row=1,
            col=2,
        )
    if config.best_model is not None:
        traces = plot_best_performance(config)
        fig.add_trace(
            traces[0],
            row=2,
            col=1,
        )
        fig.add_trace(
            traces[1],
            row=2,
            col=2,
        )
    imp_trace = get_imp_trace(config)
    fig.add_trace(
        imp_trace,
        row=3,
        col=1,
    )

    if config.best_model is not None:
        fig.add_vline(
            x=config.best_model,
            line_color='red',
            line_dash='dash',
            row=1,
            col=2,
        )
    fig.update_layout(
        width=1500,
        height=1500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        bargap=0.1,
    )
    fig.update_xaxes(
        showline=True,
        linewidth=0.5,
        linecolor='black',
        showgrid=False,
        ticks="outside",
    )
    fig.update_yaxes(
        showline=True,
        linewidth=0.5,
        linecolor='black',
        showgrid=False,
        ticks="outside",
    )
    models_df = pd.read_csv(
        f'{config.output_folder}/modelPerformance.csv'
    )
    fig.update_layout(    
        {    
            'yaxis':{'range': [0, 1], 'title_text': 'Performance'},
            'yaxis3':{'range': [-1, 1], 'title_text': 'True Delta'},
            'yaxis4':{'range': [-1, 1], 'title_text': 'True Delta'},
            'xaxis3':{'range': [-1, 1], 'title_text': 'Predicted Delta'},
            'xaxis4':{'range': [-1, 1], 'title_text': 'Predicted Delta'},
            'yaxis2':{'range': [0, ceil(models_df['modelSize']/1_048_576)]},
        }
    )
    fig.show()
    # pio.write_image(fig, f'{config.output_folder}/performanceFigures.png', engine="kaleido")
