import numpy as np
import pandas as pd
import sys
import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from .HITS import hits_algorithm
from utils import calculate_trade_distance

# 注意：为了和原来区分，我把类名也改成了 GeobiasedHITS_exp
class GeobiasedHITS_exp(hits_algorithm):
    def __init__(self, adj_matrix, coords_dict, node_names, gamma=1.0, max_iter=1000, tol=1e-8):
        """
        :param coords_dict: {name: (lat, lon)}
        """
        biased_matrix = self._apply_gravity_model(adj_matrix, coords_dict, node_names, gamma)
        super().__init__(biased_matrix, max_iter=max_iter, tol=tol)

    def _apply_gravity_model(self, adj_matrix, coords_dict, node_names, gamma):
        """
        新的指数衰减公式: W' = W * exp(-gamma * d)
        距离越大，exp的值越接近0，权重衰减越厉害。
        """
        n = adj_matrix.shape[0]
        biased_matrix = adj_matrix.copy()
                
        for i in range(n):
            for j in range(n):
                if biased_matrix[i, j] > 0:
                    u_name = node_names[i]
                    v_name = node_names[j]
                    
                    if u_name in coords_dict and v_name in coords_dict:
                        dist = calculate_trade_distance(coords_dict[u_name], coords_dict[v_name])
                        
                        # ================= 核心修改区 =================
                        # 使用 np.exp 指数函数进行衰减 (同样除以1000防止数字太大)
                        # np.exp(-gamma * dist) 是引力模型中最标准的写法
                        biased_matrix[i, j] = biased_matrix[i, j] * np.exp(-gamma * (dist / 1000))
                        # ============================================
                        
                    else:
                        print(f"找不到坐标，跳过匹配: {u_name} 或 {v_name}")
                        biased_matrix[i, j] = biased_matrix[i, j] * 0.0
                        
        return biased_matrix