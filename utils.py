import re
import math
from geopy.distance import geodesic
from difflib import get_close_matches

# Trade distance (GCD)
def calculate_trade_distance(coord1:tuple, coord2:tuple) -> float:
    return geodesic(coord1, coord2).km # Great Circle Distance

GLOBAL_NAME_MAPPING = {
    # --- Europe ---
    "Netherlands (Kingdom of the)": "Netherlands",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "Russian Federation": "Russia",
    "Czechia": "Czech Republic",
    "Republic of Moldova": "Moldova",
    "North Macedonia": "Macedonia [FYROM]",
    
    # --- Aisa ---
    "China, mainland": "China",
    "China, Hong Kong SAR": "Hong Kong",
    "China, Macao SAR": "Macau",
    "China, Taiwan Province of": "Taiwan",
    "Republic of Korea": "South Korea",
    "Democratic People's Republic of Korea": "North Korea",
    "Viet Nam": "Vietnam",
    "Lao People's Democratic Republic": "Laos",
    "Myanmar": "Burma",  # 核心修复：经纬度库通常使用旧称 Burma
    "Brunei Darussalam": "Brunei",
    "Iran (Islamic Republic of)": "Iran",
    "Syrian Arab Republic": "Syria",
    "Türkiye": "Turkey",
    "Palestine": "Palestinian Territories",
    
    # --- Africa ---
    "Democratic Republic of the Congo": "Congo [DRC]",
    "Congo": "Congo [Republic]",
    "Cabo Verde": "Cape Verde",
    "Côte d'Ivoire": "Ivory Coast",
    "Eswatini": "Swaziland",
    "United Republic of Tanzania": "Tanzania",
    "South Sudan": "Sudan", # 如果坐标库不支持南苏丹，回退到苏丹坐标
    
    # --- America ---
    "United States of America": "United States",
    "Bolivia (Plurinational State of)": "Bolivia",
    "Venezuela (Bolivarian Republic of)": "Venezuela",
    
    # --- Oceania ---
    "Micronesia (Federated States of)": "Micronesia"
}

def align_coordinates(fao_nodes, coords_df):
    """
    三重保险机制：将 FAOSTAT 节点名称自动对齐到经纬度坐标
    """
    raw_coords = {row['name']: (row['latitude'], row['longitude']) for _, row in coords_df.iterrows()}
    raw_names = list(raw_coords.keys())
    
    aligned_coords = {}
    missing = []
    
    for fao_name in fao_nodes:
        if fao_name in raw_coords:
            aligned_coords[fao_name] = raw_coords[fao_name]
            continue
            
        if fao_name in GLOBAL_NAME_MAPPING:
            mapped_name = GLOBAL_NAME_MAPPING[fao_name]
            if mapped_name in raw_coords:
                aligned_coords[fao_name] = raw_coords[mapped_name]
                continue
                
        clean_name = re.sub(r'\s*\(.*?\)\s*', '', fao_name).strip()
        if clean_name in raw_coords:
            aligned_coords[fao_name] = raw_coords[clean_name]
            continue
            
        matches = get_close_matches(clean_name, raw_names, n=1, cutoff=0.85)
        if matches:
            aligned_coords[fao_name] = raw_coords[matches[0]]
            continue
            
        missing.append(fao_name)
        
    if missing:
        print(f"\n⚠️ 提示: 全球网络中有 {len(missing)} 个微型国家/地区无法自动匹配坐标。")
        print(f"它们将被降级处理。示例: {missing[:5]}\n")
        
    return aligned_coords