from scipy.stats import kruskal, kstest, norm, ks_2samp
from src.algorithm import kruskal_wallis_custom, ks_1samp_manual, ks_2samp_manual
from src.data_processing import load_salary_data, load_game_data, load_airbnb_data, load_california_1samp, load_secom_data
import numpy as np

def run_kw_comparison(test_name, arrays):
    print(f"\n{'='*70}")
    print(f"KRUSKAL-WALLIS TEST: {test_name}")
    print(f"{'='*70}")
    H_stat, p_val, _, _, _ = kruskal_wallis_custom(*arrays)
    stat_scipy, p_scipy = kruskal(*arrays)

    print("{:<30} | {:<15} | {:<15}".format("Method", "H-statistic", "P-value"))
    print("-" * 65)
    print("{:<30} | {:<15.6f} | {:<15.6e}".format("Custom Code", H_stat, p_val))
    print("{:<30} | {:<15.6f} | {:<15.6e}".format("Scipy Library", stat_scipy, p_scipy))
    print("-" * 65)

    diff_h = abs(H_stat - stat_scipy)
    diff_p = abs(p_val - p_scipy)
    print(f"[Absolute Deviation]: H Error = {diff_h:.10f} | P-value Error = {diff_p:.10f}")

def run_ks_comparison(test_name, data):
    print(f"\n{'='*70}")
    print(f"KOLMOGOROV-SMIRNOV TEST: {test_name}")
    print(f"{'='*70}")
    std_data = (data - np.mean(data)) / np.std(data) if np.std(data) > 0 else data
    
    d_stat_manual, _, _ = ks_1samp_manual(std_data, norm.cdf)
    res_scipy = kstest(std_data, 'norm')
    d_stat_scipy = res_scipy.statistic

    print("{:<30} | {:<15}".format("Method", "D-statistic"))
    print("-" * 47)
    print("{:<30} | {:<15.6f}".format("Custom Code", d_stat_manual))
    print("{:<30} | {:<15.6f}".format("Scipy Library", d_stat_scipy))
    print("-" * 47)

    diff_d = abs(d_stat_manual - d_stat_scipy)
    print(f"[Absolute Deviation]: D Error = {diff_d:.10f}")
    
def run_ks_1samp_test():
    print(f"\n{'='*70}")
    print(f"K-S 1-SAMPLE TEST: California Housing Prices")
    print(f"{'='*70}")
    
    data = load_california_1samp()
    d_stat_manual, _, reject = ks_1samp_manual(data, norm.cdf)
    d_stat_scipy = kstest(data, 'norm').statistic
    
    print("{:<30} | {:<15} | {:<15}".format("Metric", "Value", "Comparison"))
    print("-" * 65)
    print("{:<30} | {:<15.6f} | {:<15}".format("Custom D-Statistic", d_stat_manual, "Manual"))
    print("{:<30} | {:<15.6f} | {:<15}".format("Scipy D-Statistic", d_stat_scipy, "Reference"))
    print("{:<30} | {:<15} | {:<15}".format("Reject Null?", str(reject), "Conclusion"))
    print("-" * 65)
    print(f"Absolute Deviation: {abs(d_stat_manual - d_stat_scipy):.10f}")

def run_ks_2samp_test():
    print(f"\n{'='*70}")
    print(f"K-S 2-SAMPLE TEST: SECOM Sensor #59")
    print(f"{'='*70}")
    
    sensor_good, sensor_bad = load_secom_data()
    
    if len(sensor_good) > 0 and len(sensor_bad) > 0:
        d_stat_manual, _, reject = ks_2samp_manual(sensor_good, sensor_bad)
        d_stat_scipy = ks_2samp(sensor_good, sensor_bad).statistic
        
        print("{:<30} | {:<15} | {:<15}".format("Metric", "Value", "Comparison"))
        print("-" * 65)
        print("{:<30} | {:<15.6f} | {:<15}".format("Custom D-Statistic", d_stat_manual, "Manual"))
        print("{:<30} | {:<15.6f} | {:<15}".format("Scipy D-Statistic", d_stat_scipy, "Reference"))
        print("{:<30} | {:<15} | {:<15}".format("Reject Null?", str(reject), "Conclusion"))
        print("-" * 65)
        print(f"Absolute Deviation: {abs(d_stat_manual - d_stat_scipy):.10f}")
    else:
        print("SECOM Data could not be loaded. Skipping test.")
        
def main():
    print("STARTING UNIFIED AUTOMATED TESTING SYSTEM (KS & KW)...\n")
    
    df_salary, arrays_salary, _, _ = load_salary_data('data/ds_salaries.csv')
    if df_salary is not None:
        run_kw_comparison("IT SALARY COMPARISON", arrays_salary)

    df_game, arrays_game, _, _ = load_game_data('data/vgsales.csv')
    if df_game is not None:
        run_kw_comparison("GAME SALES COMPARISON BY GENRE", arrays_game)

    df_airbnb, arrays_airbnb, _, _ = load_airbnb_data('data/AB_NYC_2019.csv')
    if df_airbnb is not None:
        run_kw_comparison("AIRBNB ROOM RENTAL PRICE COMPARISON", arrays_airbnb)
        
    run_ks_1samp_test()
    run_ks_2samp_test()
        
    print("\nALL TESTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()