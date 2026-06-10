import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
from src.algorithm import kruskal_wallis_custom, ks_1samp_manual, ks_2samp_manual, ks_2d_2samp_manual, perform_dunn_test
from src.data_processing import (
    load_salary_data, load_game_data, load_airbnb_data,
    load_california_1samp, load_secom_data, load_california_spatial
)
from src.visualization import (
    create_overlay_histogram, create_violin_plot,
    plot_ks_1samp, plot_ks_2samp, plot_ks_2d_crosshair, create_dunn_heatmap
)

st.set_page_config(page_title="Modern Non-Parametric Tests", layout="wide")

st.title("Kruskal-Wallis & Kolmogorov-Smirnov Tests")
st.markdown("---")

st.sidebar.header("Dataset Selection")
test_category = st.sidebar.selectbox(
    "Select Statistical Test Family:",
    ("Kruskal-Wallis", "Kolmogorov-Smirnov")
)

if "Kruskal-Wallis" in test_category:
    dataset_option = st.sidebar.radio(
        "Select Real-world Hypothesis:",
        ("1. IT Salary Disparity", "2. Game Sales Equivalency", "3. Airbnb Price Floor")
    )
    use_verbose = st.sidebar.checkbox("Enable Verbose Mode (Algorithm Details)")

    df, arrays, x_col, y_col = None, None, None, None

    if "IT Salary" in dataset_option:
        df, arrays, x_col, y_col = load_salary_data()
        title = "Market Compensation: Data Scientist vs Engineer vs Analyst"
    elif "Game Sales" in dataset_option:
        df, arrays, x_col, y_col = load_game_data()
        title = "Global Sales Comparison: Top 5 Game Genres"
    elif "Airbnb Price" in dataset_option:
        df, arrays, x_col, y_col = load_airbnb_data()
        title = "Price Floor Dynamics: Shared Rooms in Outer Boroughs"

    if df is not None:
        st.subheader("1. Distribution & Normality Check (Custom K-S Test)")
        unique_groups = df[x_col].unique()
        is_violating_normality = False
        ks_summary_data = []

        for group in unique_groups:
            group_data = df[df[x_col] == group][y_col].dropna().values
            if len(group_data) > 5000:
                np.random.seed(42)
                group_data = np.random.choice(group_data, 5000, replace=False)
            
            if len(group_data) > 3:
                std_dev = np.std(group_data)
                standardized_group = (group_data - np.mean(group_data)) / std_dev if std_dev > 0 else group_data
                d_stat, d_crit, reject_null = ks_1samp_manual(standardized_group, norm.cdf)
                status = "Non-Normal" if reject_null else "Normal"
                if reject_null:
                    is_violating_normality = True
                    
                ks_summary_data.append({
                    "Group Name": group,
                    "Sample Size": len(group_data),
                    "D-Statistic": f"{d_stat:.4f}",
                    "Critical Value": f"{d_crit:.4f}",
                    "Distribution": status
                })

        fig_hist = create_overlay_histogram(df, x_col, y_col, dataset_option)
        st.plotly_chart(fig_hist, use_container_width=True)
        st.dataframe(pd.DataFrame(ks_summary_data), use_container_width=True)

        if is_violating_normality:
            st.warning("**Normality Evaluation:** The Custom Kolmogorov-Smirnov test rejects normality (D-Stat > D-Crit). Proceeding to non-parametric Kruskal-Wallis.")
        else:
            st.info("**Normality Evaluation:** All individual groups satisfy the assumption of normality.")
            
        st.subheader("2. Kruskal-Wallis Test & Practical Significance (Effect Size)")
        H_stat, p_val, epsilon_sq, df_stat, details = kruskal_wallis_custom(*arrays, verbose=use_verbose)
        total_n = sum([len(a) for a in arrays])
        
        if epsilon_sq < 0.01: effect_label = "Negligible"
        elif epsilon_sq < 0.08: effect_label = "Small"
        elif epsilon_sq < 0.26: effect_label = "Moderate"
        else: effect_label = "Large"
            
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("H-Statistic", f"{H_stat:.4f}")
        col2.metric("P-value", f"{p_val:.4e}")
        col3.metric("Effect Size (ε²)", f"{epsilon_sq:.4f}")
        col4.metric("Magnitude", effect_label)
        if use_verbose:
            with st.expander("ALGORITHM EXECUTION DETAILS (VERBOSE)", expanded=True):
                st.code(f"""N = {details['N']}
N = {details['N']}
k = {details['k']}
n_i = {details['n_i']}
H_unadjusted = {details['H_unadjusted']:.4f}
correction_factor = {details['correction_factor']:.6f}
H_final = {H_stat:.4f}
Epsilon_Squared = max(0, (H_final - k + 1) / (N - k)) = {details['epsilon_sq']:.4f}""", language="python")
        st.markdown("#### Scientific Interpretation & Business Insight")
        if p_val < 0.05:
            st.success(f"**Status: Reject Null Hypothesis ($H_0$)**\n\nStatistical evidence confirms a significant difference in median values.")
        
            if epsilon_sq < 0.01:
                st.warning(f"**Insight: Statistically Significant, but Practically Negligible!**\nDue to the massive sample size (N={total_n}), the algorithm detected a microscopic variance. However, the effect size (ε² = {epsilon_sq:.4f}) is mathematically negligible. The data distributions essentially overlap.")
            elif epsilon_sq < 0.08:
                st.info(f"**Insight: Statistically Significant with a Small Effect.**\nThe Kruskal-Wallis test proves the groups are fundamentally different, but the effect size (ε² = {epsilon_sq:.4f}) indicates the magnitude of this difference is minor.")
            else:
                st.success(f"**Insight: Highly Significant (Statistically & Practically).**\nThe p-value confirms a difference, and the substantial effect size (ε² = {epsilon_sq:.4f}) proves this separation is large enough to warrant distinct strategic treatments.")
            
            st.subheader("3. Post-hoc Analysis (Dunn's Test)")
            dunn_table = perform_dunn_test(df, x_col, y_col)
            fig_heat = create_dunn_heatmap(dunn_table)
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info(f"**Status: Fail to Reject Null Hypothesis ($H_0$)**\n\nNo statistically significant differences were detected. The populations share statistically identical median values.")
            st.markdown("**Insight: Data Equivalency.**\nAny observed variance between these groups is entirely consistent with random sampling noise. There is no evidence of a systemic disparity.")
        
        
        st.subheader(f"{'4' if p_val < 0.05 else '3'}. Density Visualization")
        fig_violin = create_violin_plot(df, x_col, y_col, title, dataset_option)
        st.plotly_chart(fig_violin, use_container_width=True)
    else:
        st.error("Dataset not found. Please verify the /data directory.")

elif "Kolmogorov-Smirnov" in test_category:
    dataset_option = st.sidebar.radio(
        "Select K-S Application:",
        ("1. 1-Sample: California Housing", "2. 2-Sample: SECOM Quality Control", "3. 2D K-S Test: Spatial Segregation")
    )
    
    if "1-Sample" in dataset_option:
        st.subheader("Application 1: 1-Sample K-S Test (Normality of House Prices)")
        data = load_california_1samp()
        d_stat, d_crit, reject_null = ks_1samp_manual(data, norm.cdf)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("D-Statistic", f"{d_stat:.4f}")
        col2.metric("Critical Value", f"{d_crit:.4f}")
        col3.metric("Reject Null?", str(reject_null))
        
        fig = plot_ks_1samp(data, norm.cdf, "California Housing Prices vs. Normal Distribution")
        st.pyplot(fig)

    elif "SECOM" in dataset_option:
        st.subheader("Application 2: 2-Sample K-S Test (SECOM Sensor #59)")
        with st.spinner('Fetching SECOM data from UCI...'):
            sensor_good, sensor_bad = load_secom_data()
        d_stat, d_crit, reject_null = ks_2samp_manual(sensor_good, sensor_bad)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("D-Statistic", f"{d_stat:.4f}")
        col2.metric("Critical Value", f"{d_crit:.4f}")
        col3.metric("Reject Null?", str(reject_null))
        
        fig = plot_ks_2samp(sensor_good, sensor_bad, "Passed QC", "Failed QC", "SECOM Sensor #59 Distributions")
        st.pyplot(fig)

    elif "2D K-S" in dataset_option:
        st.subheader("Application 3: 2D Spatial K-S Test (Income Segregation)")
        high_income, low_income = load_california_spatial()
        d_stat, d_crit, reject_null, best_x, best_y = ks_2d_2samp_manual(high_income, low_income)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("2D D-Statistic", f"{d_stat:.4f}")
        col2.metric("Critical Value", f"{d_crit:.4f}")
        col3.metric("Reject Null?", str(reject_null))
        
        fig = plot_ks_2d_crosshair(high_income, low_income, "High Income", "Low Income", 
                                   "2D K-S Test: Spatial Segregation by Income", best_x, best_y, d_stat)
        st.pyplot(fig)
        
        