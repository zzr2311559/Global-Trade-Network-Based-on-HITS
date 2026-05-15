import os
import re
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULT_DIR = os.path.join(BASE_DIR, "result")
EXTERNAL_DATA_PATH = os.path.join(BASE_DIR, "evaluate", "external_alignment", "External_Metrics", "europe_lpi_2021.csv")
OUTPUT_IMG_PATH = os.path.join(os.path.dirname(__file__), "spearman_LPI_ULTIMATE_Battle_Top30.png")

def run_evaluation():
    ext_df = pd.read_csv(EXTERNAL_DATA_PATH)
    # 把四种模型的文件全抓出来
    prefixes = ['geobiased_', 'reward_ori_geobiased_', 'log10_geobiased_', 'reward_lg_geobiased_']
    result_files = [f for f in os.listdir(RESULT_DIR) if f.endswith('.csv') and any(p in f for p in prefixes)]
    
    eval_results = []
    # 只锁定最核心的 Top-30 
    K = 30 
    
    for file in result_files:
        df = pd.read_csv(os.path.join(RESULT_DIR, file))
        match = re.search(r'gamma_([0-9.]+)', file)
        gamma = float(match.group(1)) if match else None
        if gamma is None: continue
            
        if file.startswith('geobiased_'): model_type = '1. Power-Law Penalty'
        elif file.startswith('reward_ori_'): model_type = '2. Power-Law Reward'
        elif file.startswith('log10_geobiased_'): model_type = '3. Log10 Penalty'
        elif file.startswith('reward_lg_'): model_type = '4. Log10 Reward'
        else: continue

        merged_df = pd.merge(df, ext_df, on='Country', how='inner')
        temp_df = merged_df.sort_values(by='Hub Score (出)', ascending=False).head(K)
        if len(temp_df) < 5: continue
            
        coef, _ = spearmanr(temp_df['Hub Score (出)'], temp_df['LPI_Score'])
        eval_results.append({'Model': model_type, 'Gamma': gamma, 'Spearman_Coef': coef})

    results_df = pd.DataFrame(eval_results).sort_values(by=['Model', 'Gamma'])
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(11, 7))
    
    colors = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']
    markers = ['o', 'v', 's', 'D']
    
    sns.lineplot(data=results_df, x='Gamma', y='Spearman_Coef', hue='Model', style='Model',
                 markers=markers, dashes=False, linewidth=2.5, markersize=10, palette=colors)
    
    plt.title(r'Mathematical Evolution: Top-30 LPI Alignment Comparison', fontsize=16, pad=15, fontweight='bold')
    plt.xlabel(r'Distance Parameter ($\gamma$)', fontsize=13)
    plt.ylabel('Spearman Correlation (vs World Bank LPI)', fontsize=13)
    plt.xticks(np.arange(0, 1.2, 0.2))
    plt.legend(title="Algorithm Models", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_PATH, dpi=300)
    print(f"\n LPI : {OUTPUT_IMG_PATH}")

if __name__ == "__main__":
    run_evaluation()