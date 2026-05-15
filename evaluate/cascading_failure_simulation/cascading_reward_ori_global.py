import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULT_DIR = os.path.join(BASE_DIR, "result")
TRADE_DATA_PATH = os.path.join(BASE_DIR, "data", "TradeMatrix_Global", "Trade_DetailedTradeMatrix_E_All_Data_NOFLAG.csv")
if not os.path.exists(TRADE_DATA_PATH):
    TRADE_DATA_PATH = os.path.join(BASE_DIR, "data", "TradeMatrix_Global", "Trade_DetailedTradeMatrix_E_All_Data.csv")

OUTPUT_IMG_PATH = os.path.join(os.path.dirname(__file__), "cascading_failure_reward_ori_global.png")
YEAR = 2021
MAX_ATTACK_NODES = 30  
GAMMAS = [0.2, 0.4, 0.6, 0.8, 1.0]  

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
    out_degrees = {n: sum(d['weight'] for _, _, d in G.out_edges(n, data=True)) for n in G.nodes()}
    sequences['Real Export (Degree)'] = sorted(out_degrees, key=out_degrees.get, reverse=True)
    
    try:
        vanilla_df = pd.read_csv(os.path.join(RESULT_DIR, "reward_ori_vanilla_hits_scores_global_2021.csv"))
        sequences['Vanilla HITS'] = [n for n in vanilla_df.sort_values(by='Hub Score (出)', ascending=False)['Country'] if n in G]
    except: pass

    for gamma in GAMMAS:
        try:
            geo_file = f"reward_ori_geobiased_gamma_{gamma:.1f}_hits_scores_global_2021.csv"
            geo_df = pd.read_csv(os.path.join(RESULT_DIR, geo_file))
            sequences[f'Ori-Reward (γ={gamma:.1f})'] = [n for n in geo_df.sort_values(by='Hub Score (出)', ascending=False)['Country'] if n in G]
        except: pass
    return sequences

def simulate_attack(G_original, attack_sequence, total_original_weight):
    G = G_original.copy()
    original_node_count = G.number_of_nodes()
    results = {'num_removed': [0], 'volume_ratio': [1.0], 'lcc_ratio': [1.0]}
    
    for i in range(1, min(MAX_ATTACK_NODES + 1, len(attack_sequence) + 1)):
        G.remove_node(attack_sequence[i-1])
        current_weight = sum(d['weight'] for _, _, d in G.edges(data=True))
        lcc_size = len(max(nx.weakly_connected_components(G), key=len)) if G.number_of_nodes() > 0 else 0
        results['num_removed'].append(i)
        results['volume_ratio'].append(current_weight / total_original_weight)
        results['lcc_ratio'].append(lcc_size / original_node_count)
    return pd.DataFrame(results)

def run_simulation():
    G, total_weight = build_trade_network()
    sequences = get_attack_sequences(G)
    all_results = {name: simulate_attack(G, seq, total_weight) for name, seq in sequences.items()}
    plot_simulation(all_results)

def plot_simulation(all_results):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    colors = {'Real Export (Degree)': '#7f8c8d', 'Vanilla HITS': '#2c3e50'}
    markers = {'Real Export (Degree)': 'o', 'Vanilla HITS': 's'}
    geo_colors = ['#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b']
    
    for i, gamma in enumerate(GAMMAS):
        name = f'Ori-Reward (γ={gamma:.1f})'
        colors[name] = geo_colors[i]
        markers[name] = '*'

    for name, df in all_results.items():
        lw = 2.5 if 'Vanilla' in name or 'γ=1.0' in name else 1.5
        axes[0].plot(df['num_removed'], df['volume_ratio'], label=name, color=colors.get(name), marker=markers.get(name), linewidth=lw)
        axes[1].plot(df['num_removed'], df['lcc_ratio'], label=name, color=colors.get(name), marker=markers.get(name), linewidth=lw)

    axes[0].set_title('Original Reward Model: Volume Retained', fontsize=14)
    axes[1].set_title('Original Reward Model: LCC Size (Fragmentation)', fontsize=14)
    for ax in axes:
        ax.set_xlabel('Number of Hubs Removed', fontsize=12)
        ax.legend(fontsize=10)

    plt.suptitle('Global Network Robustness (Original Reward Model)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_PATH, dpi=300)
    print(f"\n 抗打击图已保存至: {OUTPUT_IMG_PATH}")

if __name__ == "__main__":
    run_simulation()