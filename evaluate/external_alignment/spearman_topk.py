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
OUTPUT_IMG_PATH = os.path.join(os.path.dirname(__file__), "spearman_topK_sensitivity_trend.png")

def run_evaluation():
    # 找刚才跑出来的 log_ 开头的文件
    result_files = [f for f in os.listdir(RESULT_DIR) if f.endswith('.csv') and 'log_geobiased' in f]
    ext_df = pd.read_csv(EXTERNAL_DATA_PATH)
    
    eval_results = []
    # 测试的不同 K 值：Top-30, Top-50, Top-100，以及 None(全部国家)
    k_list = [30, 50, 100, None] 
    
    for file in result_files:
        df = pd.read_csv(os.path.join(RESULT_DIR, file))
        match = re.search(r'gamma_([0-9.]+)', file)
        gamma = float(match.group(1)) if match else None
        if gamma is None: continue
            
        merged_df = pd.merge(df, ext_df, on='Country', how='inner')
        
        # 针对每个文件，循环计算不同的 K
        for k in k_list:
            # 如果 K 不是 None，就截取前 K 名
            if k is not None:
                temp_df = merged_df.sort_values(by='Hub Score (出)', ascending=False).head(k)
                label = f"Top-{k}"
            else:
                temp_df = merged_df
                label = "All Countries"
                
            if len(temp_df) < 5: continue
                
            coef, p_value = spearmanr(temp_df['Hub Score (出)'], temp_df['LPI_Score'])
            
            eval_results.append({
                'Gamma': gamma,
                'K_Value': label,
                'Spearman_Coef': coef
            })

    results_df = pd.DataFrame(eval_results).sort_values(by=['K_Value', 'Gamma'])
    
    # 开始画同框图
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    
    # hue='K_Value' 参数：自动用不同颜色画出不同的 K 的折线
    sns.lineplot(data=results_df, x='Gamma', y='Spearman_Coef', hue='K_Value', 
                 marker='o', linewidth=2, markersize=8, palette="Set1")
    
    plt.title(r'Sensitivity Analysis: Impact of Top-K on LPI Correlation', fontsize=15, pad=15)
    plt.xlabel(r'Distance Penalty Parameter ($\gamma$)', fontsize=12)
    plt.ylabel('Spearman Correlation (vs World Bank LPI)', fontsize=12)
    plt.xticks(np.arange(0, 1.2, 0.2))
    plt.legend(title="Country Scope", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_PATH, dpi=300)
    print(f"\n 敏感度图片已保存至: {OUTPUT_IMG_PATH}")

if __name__ == "__main__":
    run_evaluation()