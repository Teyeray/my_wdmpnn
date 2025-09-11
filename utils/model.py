import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
from torch_geometric.utils import softmax
from torch_geometric.nn import MessagePassing, global_mean_pool, AttentionalAggregation

# ------------------ Utils ------------------
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

def get_act(name: str):
    name = (name or "silu").lower()
    if name == "relu": return nn.ReLU()
    if name == "silu": return nn.SiLU()
    if name == "mish": return Mish()
    raise ValueError(f"Unknown activation {name}")

def _build_adapter(in_dim: int, out_dim: int,
                   kind: str = "linear",
                   hidden: int = 0,
                   act: str = "silu",
                   dropout: float = 0.0) -> Optional[nn.Module]:
    """
    将当前特征维度 in_dim 适配到预训练维度 out_dim。
    kind: "none" | "linear" | "mlp"
    """
    kind = (kind or "linear").lower()
    if in_dim == out_dim or kind == "none":
        return None
    if kind == "linear":
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Dropout(dropout),
        )
    if kind == "mlp":
        h = hidden if hidden and hidden > 0 else max(16, min(256, (in_dim + out_dim)//2))
        return nn.Sequential(
            nn.Linear(in_dim, h),
            get_act(act),
            nn.Dropout(dropout),
            nn.Linear(h, out_dim),
            nn.Dropout(dropout),
        )
    raise ValueError(f"Unknown adapter kind: {kind}")

# ------------------ WD-MPNN Layer ------------------
class WDMPNNLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim,
                 use_edge_attn=False, att_hidden=64,
                 dropout=0.1, act="mish"):
        super().__init__(aggr="add")
        self.use_edge_attn = use_edge_attn
        self.lin_node = nn.Linear(node_dim, hidden_dim)
        self.lin_edge = nn.Linear(edge_dim, hidden_dim)
        self.lin_update = nn.Linear(hidden_dim, hidden_dim)
        self.act = get_act(act)
        self.dropout = nn.Dropout(dropout)

        if use_edge_attn:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, att_hidden),
                nn.SiLU(),
                nn.Linear(att_hidden, 1),
            )
            self.att_dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        # x, edge_attr 已经是“进入本层前”的维度（第一层可能是 adapter 之后的维度）
        h_node = self.lin_node(x)           # [N, H]
        h_edge = self.lin_edge(edge_attr)   # [E, H]
        return self.propagate(edge_index, x=h_node, edge_attr=h_edge)

    def message(self, x_j, x_i, edge_attr, index):
        # x_j: 源节点嵌入（已线性到 H），x_i: 目标节点嵌入
        msg = x_j + edge_attr               # [E, H]
        if self.use_edge_attn:
            # 基于 (msg, x_i) 计算每条入边的注意力分数
            score = self.att_mlp(torch.cat([msg, x_i], dim=-1)).squeeze(-1)  # [E]
            alpha = softmax(score, index)  # 对同一 index(=dst) 的入边做 softmax
            alpha = self.att_dropout(alpha)
            msg = msg * alpha.unsqueeze(-1)  # [E, H]
        return msg

    def update(self, aggr_out):
        out = self.lin_update(aggr_out)
        out = self.act(out)
        return self.dropout(out)

# ------------------ Encoder（带 Adapter） ------------------
class WDMPNNEncoder(nn.Module):
    """
    支持 adapter：
      - cur_node_dim/cur_edge_dim: 当前数据集输入维度
      - pre_node_dim/pre_edge_dim: 预训练时维度（目标维度）
      - adapter_*: 适配器结构
    如果 cur_* 与 pre_* 相同，则不创建 adapter。
    """
    def __init__(self,
                 # 当前数据集的维度（直接来自 Data.x / Data.edge_attr）
                 cur_node_dim: int,
                 cur_edge_dim: int,
                 # 预训练维度（要对齐到的维度）
                 pre_node_dim: Optional[int] = None,
                 pre_edge_dim: Optional[int] = None,
                 # 消息传递结构
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 use_edge_attn: bool = False,
                 att_hidden: int = 64,
                 dropout: float = 0.1,
                 act: str = "mish",
                 pool: str = "att",
                 # 适配器配置
                 adapter_kind: str = "linear",    # "none"|"linear"|"mlp"
                 adapter_hidden: int = 0,
                 adapter_act: str = "silu",
                 adapter_dropout: float = 0.0):
        super().__init__()

        self.cur_node_dim = int(cur_node_dim)
        self.cur_edge_dim = int(cur_edge_dim)
        self.pre_node_dim = int(pre_node_dim) if pre_node_dim is not None else self.cur_node_dim
        self.pre_edge_dim = int(pre_edge_dim) if pre_edge_dim is not None else self.cur_edge_dim

        # ---- adapters（可选）----
        self.node_adapter = _build_adapter(self.cur_node_dim, self.pre_node_dim,
                                           kind=adapter_kind,
                                           hidden=adapter_hidden,
                                           act=adapter_act,
                                           dropout=adapter_dropout)
        self.edge_adapter = _build_adapter(self.cur_edge_dim, self.pre_edge_dim,
                                           kind=adapter_kind,
                                           hidden=adapter_hidden,
                                           act=adapter_act,
                                           dropout=adapter_dropout)

        # ---- MPNN layers ----
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_node_dim = self.pre_node_dim if i == 0 else hidden_dim
            in_edge_dim = self.pre_edge_dim if i == 0 else hidden_dim
            self.layers.append(WDMPNNLayer(
                node_dim=in_node_dim,
                edge_dim=in_edge_dim,
                hidden_dim=hidden_dim,
                use_edge_attn=use_edge_attn,
                att_hidden=att_hidden,
                dropout=dropout,
                act=act,
            ))

        # ---- graph pooling ----
        pool = (pool or "att").lower()
        if pool == "mean":
            self.pool_kind = "mean"
            self.pool = global_mean_pool  # function
        elif pool == "att":
            self.pool_kind = "att"
            self.pool = AttentionalAggregation(
                gate_nn=nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 1),
                )
            )
        else:
            raise ValueError(f"Unknown pool {pool}")

    def forward(self, x, edge_index, edge_attr, batch):
        # 1) 适配到预训练维度（若需要）
        if self.node_adapter is not None:
            x = self.node_adapter(x)              # [N, pre_node_dim]
        if self.edge_adapter is not None:
            edge_attr = self.edge_adapter(edge_attr)  # [E, pre_edge_dim]

        # 2) 消息传递
        h_node = x
        h_edge = edge_attr
        for i, layer in enumerate(self.layers):
            h_node = layer(h_node, edge_index, h_edge)   # 返回的是新的 node 隐状态
            # 下一层的 edge 输入使用上一层的边映射：这里用上一层的 lin_edge 再处理一次更一致
            # 但我们的实现中 layer.forward 内部会每层都先 lin_node/lin_edge，
            # 因此继续沿用原始 edge_attr（或你也可以选择 h_edge = layer.lin_edge(edge_attr)）
            h_edge = h_node.new_zeros(edge_attr.size(0), h_node.size(-1))  # 让后续层的edge线性层接手

        # 3) 池化
        if self.pool_kind == "mean":
            graph_repr = self.pool(h_node, batch)
        else:
            graph_repr = self.pool(h_node, batch)
        return graph_repr

    # 方便外部单独操作 adapter 参数
    def adapter_parameters(self):
        params = []
        if self.node_adapter is not None:
            params += list(self.node_adapter.parameters())
        if self.edge_adapter is not None:
            params += list(self.edge_adapter.parameters())
        return params

    def has_adapter(self) -> bool:
        return (self.node_adapter is not None) or (self.edge_adapter is not None)

# ------------------ Task Head ------------------
class MultiTaskHead(nn.Module):
    """
    - tasks=None: 单任务 → 输出 shape [B]
    - tasks=List[str]: 多任务 → 输出字典 {task: [B]}
    """
    def __init__(self, hidden_dim, mlp_hidden=[128, 64], tasks=None,
                 dropout=0.1, act="silu"):
        super().__init__()
        self.tasks = tasks
        self.act = get_act(act)

        layers = []
        cur = hidden_dim
        for h in mlp_hidden:
            layers += [nn.Linear(cur, h), self.act, nn.Dropout(dropout)]
            cur = h
        out_dim = 1 if tasks is None else len(tasks)
        layers.append(nn.Linear(cur, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, graph_repr):
        y = self.mlp(graph_repr)  # [B,1] 或 [B,T]
        if self.tasks is None:
            return y.squeeze(-1)
        return {t: y[:, i] for i, t in enumerate(self.tasks)}

# ------------------ Full Model（模块可独立保存/加载/冻结） ------------------
class WDMPNNModel(nn.Module):
    def __init__(self,
                 # 当前数据集的输入维度
                 node_dim, edge_dim,
                 # 预训练（目标）维度：若不填，则与当前相同=不使用adapter
                 pre_node_dim: Optional[int] = None,
                 pre_edge_dim: Optional[int] = None,
                 # 编码器结构
                 hidden_dim=256, num_layers=3,
                 use_edge_attn=True, dropout=0.1, act="mish", pool="att",
                 # 适配器
                 adapter_kind="linear", adapter_hidden=0, adapter_act="silu", adapter_dropout=0.0,
                 # 任务头
                 tasks=None, mlp_hidden=[128, 64]):
        super().__init__()
        self.encoder = WDMPNNEncoder(
            cur_node_dim=node_dim,
            cur_edge_dim=edge_dim,
            pre_node_dim=pre_node_dim,
            pre_edge_dim=pre_edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_edge_attn=use_edge_attn,
            att_hidden=64,
            dropout=dropout,
            act=act,
            pool=pool,
            adapter_kind=adapter_kind,
            adapter_hidden=adapter_hidden,
            adapter_act=adapter_act,
            adapter_dropout=adapter_dropout,
        )
        self.head = MultiTaskHead(hidden_dim, mlp_hidden=mlp_hidden,
                                  tasks=tasks, dropout=dropout, act=act)

    # -------- 前向 --------
    def forward(self, data):
        g = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        return self.head(g)

    # -------- 冻结/解冻 --------
    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    def freeze_adapter(self):
        for p in self.encoder.adapter_parameters():
            p.requires_grad = False

    def unfreeze_adapter(self):
        for p in self.encoder.adapter_parameters():
            p.requires_grad = True

    def freeze_head(self):
        for p in self.head.parameters():
            p.requires_grad = False

    def unfreeze_head(self):
        for p in self.head.parameters():
            p.requires_grad = True

    # -------- 参数分组（便于设不同学习率）--------
    def param_groups(self) -> Dict[str, List[nn.Parameter]]:
        groups = {"encoder": [], "adapter": [], "head": []}
        # adapter
        for p in self.encoder.adapter_parameters():
            if p.requires_grad:
                groups["adapter"].append(p)
        # encoder（排除 adapter）
        enc_prefix = {"node_adapter", "edge_adapter"}
        for name, p in self.encoder.named_parameters():
            if not p.requires_grad:
                continue
            if any(name.startswith(pref) for pref in enc_prefix):
                continue
            groups["encoder"].append(p)
        # head
        for p in self.head.parameters():
            if p.requires_grad:
                groups["head"].append(p)
        return groups

    # -------- 保存/加载：分别对 encoder / adapter / head / full --------
    # 单独保存/加载 encoder（不含 adapter）
    def save_encoder(self, path: str):
        # 过滤掉 adapter 参数
        state = {k: v for k, v in self.encoder.state_dict().items()
                 if not (k.startswith("node_adapter") or k.startswith("edge_adapter"))}
        torch.save(state, path)

    def load_encoder(self, path: str, strict: bool = True, map_location="cpu"):
        state = torch.load(path, map_location=map_location)
        missing, unexpected = self.encoder.load_state_dict(state, strict=strict)
        if not strict and (missing or unexpected):
            print("[load_encoder] missing:", missing, "unexpected:", unexpected)

    # 单独保存/加载 adapter
    def save_adapter(self, path: str):
        ad_state = {}
        if self.encoder.node_adapter is not None:
            ad_state["node_adapter"] = self.encoder.node_adapter.state_dict()
        if self.encoder.edge_adapter is not None:
            ad_state["edge_adapter"] = self.encoder.edge_adapter.state_dict()
        torch.save(ad_state, path)

    def load_adapter(self, path: str, strict: bool = True, map_location="cpu"):
        ad_state = torch.load(path, map_location=map_location)
        if "node_adapter" in ad_state and self.encoder.node_adapter is not None:
            self.encoder.node_adapter.load_state_dict(ad_state["node_adapter"], strict=strict)
        if "edge_adapter" in ad_state and self.encoder.edge_adapter is not None:
            self.encoder.edge_adapter.load_state_dict(ad_state["edge_adapter"], strict=strict)

    # 单独保存/加载 head
    def save_head(self, path: str):
        torch.save(self.head.state_dict(), path)

    def load_head(self, path: str, strict: bool = True, map_location="cpu"):
        state = torch.load(path, map_location=map_location)
        missing, unexpected = self.head.load_state_dict(state, strict=strict)
        if not strict and (missing or unexpected):
            print("[load_head] missing:", missing, "unexpected:", unexpected)

    # 整体保存/加载
    def save_full(self, path: str):
        torch.save(self.state_dict(), path)

    def load_full(self, path: str, strict: bool = True, map_location="cpu"):
        state = torch.load(path, map_location=map_location)
        missing, unexpected = self.load_state_dict(state, strict=strict)
        if not strict and (missing or unexpected):
            print("[load_full] missing:", missing, "unexpected:", unexpected)



# ------------------ Traditional ML Model Configurations ------------------
class MLModelConfig:
    """Configuration class for traditional ML models"""
    def __init__(self):
        # Data processing
        self.random_state = 42
        self.n_folds = 5
        self.use_stratified_cv = True
        
        # Feature processing
        self.use_variance_threshold = True
        self.variance_threshold = 0.01
        self.use_correlation_filter = True
        self.correlation_threshold = 0.95
        self.use_feature_selection = True
        self.use_robust_scaler = True
        
        # Model parameters by task
        self.xgb_params = {
            'Tg': {
                'n_estimators': 3000, 'max_depth': 5, 'learning_rate': 0.01,
                'subsample': 0.8, 'colsample_bytree': 1.0, 'reg_lambda': 7.0,
                'gamma': 0.1, 'objective': 'reg:absoluteerror', 'eval_metric': 'mae',
                'early_stopping_rounds': 50, 'random_state': 42
            },
            'FFV': {
                'n_estimators': 3000, 'max_depth': 7, 'learning_rate': 0.06,
                'subsample': 0.6, 'colsample_bytree': 0.8, 'reg_lambda': 2.0,
                'gamma': 0.0, 'objective': 'reg:absoluteerror', 'eval_metric': 'mae',
                'early_stopping_rounds': 50, 'random_state': 42
            },
            'Tc': {
                'n_estimators': 3000, 'max_depth': 4, 'learning_rate': 0.01,
                'subsample': 0.6, 'colsample_bytree': 0.8, 'reg_lambda': 7.0,
                'gamma': 0.0, 'objective': 'reg:absoluteerror', 'eval_metric': 'mae',
                'early_stopping_rounds': 50, 'random_state': 42
            },
            'Density': {
                'n_estimators': 3000, 'max_depth': 5, 'learning_rate': 0.06,
                'subsample': 0.8, 'colsample_bytree': 1.0, 'reg_lambda': 3.0,
                'gamma': 0.0, 'objective': 'reg:absoluteerror', 'eval_metric': 'mae',
                'early_stopping_rounds': 50, 'random_state': 42
            },
            'Rg': {
                'n_estimators': 3000, 'max_depth': 4, 'learning_rate': 0.06,
                'subsample': 0.6, 'colsample_bytree': 1.0, 'reg_lambda': 10.0,
                'gamma': 0.1, 'objective': 'reg:absoluteerror', 'eval_metric': 'mae',
                'early_stopping_rounds': 50, 'random_state': 42
            }
        }
        
        # LightGBM parameters
        self.lgb_params = {
            'Tg': {
                'n_estimators': 3000, 'max_depth': 5, 'learning_rate': 0.01,
                'subsample': 0.8, 'colsample_bytree': 1.0, 'reg_lambda': 7.0,
                'objective': 'mae', 'metric': 'mae', 'verbosity': -1,
                'random_state': 42, 'force_col_wise': True
            },
            'FFV': {
                'n_estimators': 3000, 'max_depth': 7, 'learning_rate': 0.06,
                'subsample': 0.6, 'colsample_bytree': 0.8, 'reg_lambda': 2.0,
                'objective': 'mae', 'metric': 'mae', 'verbosity': -1,
                'random_state': 42, 'force_col_wise': True
            },
            'Tc': {
                'n_estimators': 3000, 'max_depth': 4, 'learning_rate': 0.01,
                'subsample': 0.6, 'colsample_bytree': 0.8, 'reg_lambda': 7.0,
                'objective': 'mae', 'metric': 'mae', 'verbosity': -1,
                'random_state': 42, 'force_col_wise': True
            },
            'Density': {
                'n_estimators': 3000, 'max_depth': 5, 'learning_rate': 0.06,
                'subsample': 0.8, 'colsample_bytree': 1.0, 'reg_lambda': 3.0,
                'objective': 'mae', 'metric': 'mae', 'verbosity': -1,
                'random_state': 42, 'force_col_wise': True
            },
            'Rg': {
                'n_estimators': 3000, 'max_depth': 4, 'learning_rate': 0.06,
                'subsample': 0.6, 'colsample_bytree': 1.0, 'reg_lambda': 10.0,
                'objective': 'mae', 'metric': 'mae', 'verbosity': -1,
                'random_state': 42, 'force_col_wise': True
            }
        }
        
        # CatBoost parameters
        self.cat_params = {
            'Tg': {
                'iterations': 3000, 'depth': 5, 'learning_rate': 0.01,
                'l2_leaf_reg': 7.0, 'loss_function': 'MAE',
                'eval_metric': 'MAE', 'random_seed': 42, 'verbose': False,
                'early_stopping_rounds': 50
            },
            'FFV': {
                'iterations': 3000, 'depth': 7, 'learning_rate': 0.06,
                'l2_leaf_reg': 2.0, 'loss_function': 'MAE',
                'eval_metric': 'MAE', 'random_seed': 42, 'verbose': False,
                'early_stopping_rounds': 50
            },
            'Tc': {
                'iterations': 3000, 'depth': 4, 'learning_rate': 0.01,
                'l2_leaf_reg': 7.0, 'loss_function': 'MAE',
                'eval_metric': 'MAE', 'random_seed': 42, 'verbose': False,
                'early_stopping_rounds': 50
            },
            'Density': {
                'iterations': 3000, 'depth': 5, 'learning_rate': 0.06,
                'l2_leaf_reg': 3.0, 'loss_function': 'MAE',
                'eval_metric': 'MAE', 'random_seed': 42, 'verbose': False,
                'early_stopping_rounds': 50
            },
            'Rg': {
                'iterations': 3000, 'depth': 4, 'learning_rate': 0.06,
                'l2_leaf_reg': 10.0, 'loss_function': 'MAE',
                'eval_metric': 'MAE', 'random_seed': 42, 'verbose': False,
                'early_stopping_rounds': 50
            }
        }


def create_ml_model(model_type: str, target: str, config: MLModelConfig):
    """Create ML model instance based on type and target"""
    if model_type.lower() == 'xgboost':
        import xgboost as xgb
        params = config.xgb_params.get(target, config.xgb_params['Tg'])
        return xgb.XGBRegressor(**params, n_jobs=-1, verbosity=0)
    
    elif model_type.lower() == 'lightgbm':
        import lightgbm as lgb
        params = config.lgb_params.get(target, config.lgb_params['Tg'])
        return lgb.LGBMRegressor(**params, n_jobs=-1)
    
    elif model_type.lower() == 'catboost':
        import catboost as cb
        params = config.cat_params.get(target, config.cat_params['Tg'])
        return cb.CatBoostRegressor(**params, thread_count=-1)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")