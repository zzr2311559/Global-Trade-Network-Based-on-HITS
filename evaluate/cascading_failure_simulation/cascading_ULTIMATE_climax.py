import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULT_DIR = os.path.join(BASE_DIR, "result")
TRADE_DATA_PATH = os.path.join(BASE_DIR, "data", "TradeMatrix_Global", "Trade_DetailedTradeMatrix_E_All_Data_NOFLAG.csv")
if not os.path.exists(TRADE_DATA_PATH):
    TRADE_DATA_PATH = os.path.join(BASE_DIR, "data", "TradeMatrix_Global", "Trade_DetailedTradeMatrix_E_All_Data.csv")

OUTPUT_IMG_PATH = os.path.join(os.path.dirname(__file__), "cascading_failure_ULTIMATE_climax.png")

YEAR = 2021
MAX_ATTACK_NODES = 30  

def build_trade_network():
    df = pd.read_csv(TRADE_DATA_PATH, low_memory=False)
    year_col = f'Y{YEAR}'
    df_trade = df[df['Element'].str.contains('Export Value', case=False, na=False)].dropna(subset=[year_col])
    G = nx.DiGraph()
    for _, row in df_trade.iterrows():
        if row[year_col] > 0:
            G.add_edge(row['Reporter Countries'], row['Partner Countries'], weight=row[year_col])
    return G, sum(d['weight'] for _, _, d in G.edges(data=True))

def get_attack_sequences(G):
    sequences = {}
    
    # 1. 传统无偏置 (Baseline)
    try:
        vanilla_df = pd.read_csv(os.path.join(RESULT_DIR, "log10_vanilla_hits_scores_global_2021.csv"))
        sequences['Baseline (Vanilla HITS)'] = [n for n in vanilla_df.sort_values(by='Hub Score (出)', ascending=False)['Country'] if n in G]
    except: pass

    # 2. 对数惩罚最强态 (寻找区域密集节点)
    try:
        geo_df = pd.read_csv(os.path.join(RESULT_DIR, "log10_geobiased_gamma_1.0_hits_scores_global_2021.csv"))
        sequences['Penalty Geo (Regional Hubs)'] = [n for n in geo_df.sort_values(by='Hub Score (出)', ascending=False)['Country'] if n in G]
    except: pass
    
    # 3. 对数奖励最强态 (寻找洲际桥梁)
    try:
        reward_df = pd.read_csv(os.path.join(RESULT_DIR, "reward_lg_geobiased_gamma_1.0_hits_scores_global_2021.csv"))
        sequences['Reward Geo (Global Anchors)'] = [n for n in reward_df.sort_values(by='Hub Score (出)', ascending=False)['Country'] if n in G]
    except: pass

    return sequences

def simulate_attack(G_original, attack_sequence, total_original_weight):
    G = G_original.copy()
    original_node_count = G.number_of_nodes()
    results = {'num_removed': [0], 'lcc_ratio': [1.0]}
    
    for i in range(1, min(MAX_ATTACK_NODES + 1, len(attack_sequence) + 1)):
        G.remove_node(attack_sequence[i-1])
        lcc_size = len(max(nx.weakly_connected_components(G), key=len)) if G.number_of_nodes() > 0 else 0
        results['num_removed'].append(i)
        results['lcc_ratio'].append(lcc_size / original_node_count)
    return pd.DataFrame(results)

def run_simulation():
    G, total_weight = build_trade_network()
    sequences = get_attack_sequences(G)
    all_results = {name: simulate_attack(G, seq, total_weight) for name, seq in sequences.items()}
    plot_simulation(all_results)

def plot_simulation(all_results):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 7))
    
    colors = {
        'Baseline (Vanilla HITS)': '#7f8c8d', 
        'Penalty Geo (Regional Hubs)': '#8e44ad',
        'Reward Geo (Global Anchors)': '#e74c3c'
    }
    markers = {'Baseline (Vanilla HITS)': 'o', 'Penalty Geo (Regional Hubs)': 's', 'Reward Geo (Global Anchors)': 'D'}
    
    for name, df in all_results.items():
        plt.plot(df['num_removed'], df['lcc_ratio'], label=name, color=colors.get(name), 
                 marker=markers.get(name), linewidth=3, markersize=9)

    plt.title('Ultimate Robustness: Fragmentation Speed Comparison ($\gamma=1.0$)', fontsize=16, pad=15, fontweight='bold')
    plt.xlabel('Number of Hubs Removed (Targeted Attack)', fontsize=13)
    plt.ylabel('Network Integration (LCC Ratio)', fontsize=13)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_PATH, dpi=300)
    print(f"\n 抗打击图已保存至: {OUTPUT_IMG_PATH}")

if __name__ == "__main__":
    run_simulation()