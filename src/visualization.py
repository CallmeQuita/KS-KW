import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import contextily as ctx

def create_dunn_heatmap(p_values_matrix):
    custom_colorscale = [
        [0.0, 'darkgreen'], 
        [0.049, 'limegreen'], 
        [0.05, 'lightcoral'], 
        [1.0, 'darkred']
    ]
    
    fig = px.imshow(
        p_values_matrix, 
        text_auto=".3f", 
        color_continuous_scale=custom_colorscale, 
        range_color=[0, 1],
        title="Dunn's Post-hoc Test - Adjusted p-values (Green = Significant)"
    )
    fig.update_layout(xaxis_title="Groups", yaxis_title="Groups", coloraxis_colorbar=dict(title="P-value"))
    return fig

def create_overlay_histogram(df, x_col, y_col, dataset_option):
    fig_hist = px.histogram(
        df, x=y_col, nbins=50, color=x_col, barmode="overlay", marginal="box",
        title=f"Overlay Histogram of {y_col} across Groups"
    )
    if "Airbnb" in dataset_option:
        fig_hist.update_layout(xaxis=dict(range=[0, 150]))
    elif "Game" in dataset_option:
        fig_hist.update_layout(xaxis=dict(range=[0, 5]))
    return fig_hist

def create_violin_plot(df, x_col, y_col, title, dataset_option):
    fig_violin = px.violin(
        df, y=x_col, x=y_col, box=True, points=False, color=x_col, 
        title=title, height=600
    )
    if "Airbnb" in dataset_option:
        fig_violin.update_layout(xaxis=dict(range=[0, 150]))
    elif "Game" in dataset_option:
        fig_violin.update_layout(xaxis=dict(range=[0, 5]))
    return fig_violin

def plot_ks_1samp(data, theoretical_cdf_func, title):

    data = data[~np.isnan(data)]
    data_sorted = np.sort(data)
    n = len(data_sorted)

    ecdf = np.arange(1, n + 1) / n
    t_cdf = theoretical_cdf_func(data_sorted)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(data_sorted, ecdf, label='Empirical Data CDF', where='post', linewidth=2, color='#1f77b4')
    ax.plot(data_sorted, t_cdf, label='Theoretical CDF (Normal)', linestyle='--', color='#ff7f0e', linewidth=2)

    diffs = np.abs(ecdf - t_cdf)
    idx_max = np.argmax(diffs)
    x_max = data_sorted[idx_max]

    ax.vlines(x_max, min(ecdf[idx_max], t_cdf[idx_max]), max(ecdf[idx_max], t_cdf[idx_max]),
               color='black', linestyle=':', linewidth=2, label=f'D-Statistic: {diffs[idx_max]:.4f}')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Standardized Value', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    return fig

def plot_ks_2samp(group_a, group_b, label_a, label_b, title):
    group_a = group_a[~np.isnan(group_a)]
    group_b = group_b[~np.isnan(group_b)]

    data_a_sorted = np.sort(group_a)
    data_b_sorted = np.sort(group_b)

    ecdf_a = np.arange(1, len(data_a_sorted) + 1) / len(data_a_sorted)
    ecdf_b = np.arange(1, len(data_b_sorted) + 1) / len(data_b_sorted)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(data_a_sorted, ecdf_a, label=label_a, where='post', linewidth=2, color='#1f77b4')
    ax.step(data_b_sorted, ecdf_b, label=label_b, where='post', linewidth=2, color='#ff7f0e')

    data_all = np.sort(np.concatenate([group_a, group_b]))
    cdf1 = np.searchsorted(data_a_sorted, data_all, side='right') / len(group_a)
    cdf2 = np.searchsorted(data_b_sorted, data_all, side='right') / len(group_b)
    diffs = np.abs(cdf1 - cdf2)

    idx_max = np.argmax(diffs)
    x_max = data_all[idx_max]
    y1_max = cdf1[idx_max]
    y2_max = cdf2[idx_max]

    ax.vlines(x_max, min(y1_max, y2_max), max(y1_max, y2_max),
               color='black', linestyle=':', linewidth=2, label=f'D-Statistic: {diffs[idx_max]:.4f}')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    return fig

def plot_ks_2d_crosshair(data1, data2, label1, label2, title, best_x, best_y, max_d_stat):
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(data1[:, 0], data1[:, 1], color='#1f77b4', alpha=0.6, label=label1, marker='o', edgecolors='w', s=50)
    ax.scatter(data2[:, 0], data2[:, 1], color='#ff7f0e', alpha=0.6, label=label2, marker='^', edgecolors='w', s=50)

    ax.axvline(best_x, color='black', linestyle='--', linewidth=2)
    ax.axhline(best_y, color='black', linestyle='--', linewidth=2)

    ax.plot(best_x, best_y, marker='s', color='red', markersize=10, fillstyle='none', markeredgewidth=3,
             label=f'Max D-Stat Origin\nD = {max_d_stat:.4f}')

    ax.text(best_x - 0.05, best_y + 0.05, 'Top-Left', fontsize=12, fontweight='bold', alpha=0.8, ha='right')
    ax.text(best_x + 0.05, best_y + 0.05, 'Top-Right', fontsize=12, fontweight='bold', alpha=0.8, ha='left')
    ax.text(best_x - 0.05, best_y - 0.05, 'Bottom-Left', fontsize=12, fontweight='bold', alpha=0.8, ha='right')
    ax.text(best_x + 0.05, best_y - 0.05, 'Bottom-Right', fontsize=12, fontweight='bold', alpha=0.8, ha='left')

    ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.CartoDB.Positron)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.legend(loc='upper right', framealpha=1)
    return fig