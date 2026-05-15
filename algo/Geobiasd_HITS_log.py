import numpy as np
from .HITS import hits_algorithm
from utils import calculate_trade_distance

class GeobiasedHITS_log(hits_algorithm):
    def __init__(self, adj_matrix, coords_dict, node_names, gamma=1.0, max_iter=1000, tol=1e-8):
        biased_matrix = self._apply_gravity_model(adj_matrix, coords_dict, node_names, gamma)
        super().__init__(biased_matrix, max_iter=max_iter, tol=tol)

    def _apply_gravity_model(self, adj_matrix, coords_dict, node_names, gamma):
        n = adj_matrix.shape[0]
        biased_matrix = adj_matrix.copy()
                
        for i in range(n):
            for j in range(n):
                if biased_matrix[i, j] > 0:
                    u_name, v_name = node_names[i], node_names[j]
                    if u_name in coords_dict and v_name in coords_dict:
                        dist = calculate_trade_distance(coords_dict[u_name], coords_dict[v_name])
                        
                        # ================= 核心修改区 =================
                        # 使用对数衰减： np.log1p(x) 相当于 ln(1+x)
                        # 这个公式比指数温和得多，惩罚不会过重
                        penalty = (np.log1p(dist / 1000) + 1) ** gamma
                        biased_matrix[i, j] = biased_matrix[i, j] / penalty
                        # ============================================
                    else:
                        biased_matrix[i, j] = 0.0
                        
        return biased_matrix