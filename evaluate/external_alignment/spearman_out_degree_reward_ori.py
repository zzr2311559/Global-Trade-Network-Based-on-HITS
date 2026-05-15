import os
import re
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULT_DIR = os.path.join(BASE_DIR, "result")
TRADE_DATA_PATH = os.path.join(BASE_DIR, "data", "TradeMatrix_Global", "Trade_DetailedTradeMatrix_E_All_Data_NOFLAG.csv")
if not os.path.exists(TRADE_DATA_PATH):
    TRADE_DATA_PATH = os.path.join(BASE_DIR, "data", "TradeMatrix_Global", "Trade_DetailedTradeMatrix_E_All_Data.csv")

OUTPUT_IMG_PATH = os.path.join(os.path.dirname(__file__), "spearman_out_degree_reward_ori_trend.png")

def calculate_real_out_degree(year=2021):
    df = pd.read_csv(TRADE_DATA_PATH, low_memory=False)
    year_col = f'Y{year}'
    df_export = df[df['Element'].str.contains('Export Value', case=False, na=False)].copy()
    out_degree_df = df_export.groupby('Reporter Countries')[year_col].sum().reset_index()
    out_degree_df.columns = ['Country', 'Real_Out_Degree']
    return out_degree_df

def run_evaluation():
    out_degree_metric = calculate_real_out_degree(year=2021)
    result_files = [f for f in os.listdir(RESULT_DIR) if f.endswith('.csv') and 'reward_ori_' in f]
    
    eval_results = []
    for file in result_files:
        df = pd.read_csv(os.path.join(RESULT_DIR, file))
        if 'vanilla' in file:
            gamma = 0.0
            model_type = 'Vanilla HITS'
        else:
            match = re.search(r'gamma_([0-9.]+)', file)
            gamma = float(match.group(1)) if match else None
            model_type = 'Original Reward Model'
            
        if gamma is None: continue
        merged = pd.merge(df, out_degree_metric, on='Country', how='inner')
        coef, _ = spearmanr(merged['Hub Score (出)'], merged['Real_Out_Degree'])
        eval_results.append({'Model': model_type, 'Gamma': gamma, 'Spearman_Coef': coef})

    results_df = pd.DataFrame(eval_results).sort_values(by='Gamma')
    plot_results(results_df)

def plot_results(results_df):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    
    geo_df = results_df[results_df['Model'] == 'Original Reward Model']
    sns.lineplot(data=geo_df, x='Gamma', y='Spearman_Coef', marker='o', 
                 linewidth=2.5, markersize=9, color='#d35400', label='Original Reward Model')
    
    vanilla_row = results_df[results_df['Model'] == 'Vanilla HITS']
    if not vanilla_row.empty:
        vanilla_coef = vanilla_row.iloc[0]['Spearman_Coef']
        plt.axhline(y=vanilla_coef, color='#2c3e50', linestyle='--', linewidth=2, label='Vanilla HITS (Baseline)')

    plt.title('Original Reward Model: Hub Score vs. Raw Export Volume', fontsize=15, pad=15)
    plt.xlabel('Distance Reward Parameter (Gamma)', fontsize=12)
    plt.ylabel('Spearman Correlation (with Total Export)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_PATH, dpi=300)
    print(f"\n 出度解耦图已保存至: {OUTPUT_IMG_PATH}")

if __name__ == "__main__":
    run_evaluation()