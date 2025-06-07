# cinema_dashboard_storyboard_final.py
"""
Village Cinemas F&B Recommender App

An interactive Streamlit dashboard designed to analyze food and beverage sales data
to uncover purchasing patterns and generate actionable recommendations.

Core Features:
1.  **Dynamic Recommendations:** Utilizes a hybrid approach combining Association
    Rule Mining (FP-Growth) and Collaborative Filtering (ALS) to suggest
    complementary items.
2.  **Contextual Filtering:** Allows users to slice the data by various segments
    (Genre, Time Slot, Language, etc.) to generate context-specific recommendations.
3.  **Recency Weighting:** Optionally prioritizes recent sales data to ensure
    recommendations are timely and relevant.
4.  **Performance Evaluation:** Measures the quality of recommendations against a
    holdout (test) dataset using standard industry metrics like Hit Rate,
    Precision, and NDCG.
5.  **Scenario Analysis:** Provides a 'What If' explorer to compare recommendations
    under two different filter contexts side-by-side.

Data Requirements:
-   All input files must be located in the 'input_datasets/' subdirectory.
-   `train_dataset.xlsx`: The primary training data.
-   `test_dataset.xlsx`: The holdout data for evaluation.
-   `inventory_transactions_clean.xlsx`: Used to map product SKUs to item classes.
"""

# ───────────────────────  PAGE CONFIG  ───────────────────────
# Must be the first Streamlit command in the script.

import streamlit as st
st.set_page_config(
    page_title="Village Cinemas F&B Recommender App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ───────────────────────  CORE IMPORTS  ─────────────────────

import hashlib
import json
import pickle
import warnings
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Final

import altair as alt
import implicit
import numpy as np
import pandas as pd
import scipy.sparse as sp
from mlxtend.frequent_patterns import association_rules, fpgrowth

# ───────────────────────  CONSTANTS & CONFIGURATION  ────────────────────────
# Centralized configuration for file paths, model parameters, and business rules.

# --- Versioning ---
# Increment this version string to invalidate all persistent caches on application restart.
CACHE_VER = "v1.1"

# --- File & Directory Paths ---
# Defines where the application looks for input data and stores cached models.
INPUTS_DIR: Final[Path] = Path("input_datasets/")
TRAINING_PATH: Final[Path]  = INPUTS_DIR / "train_dataset.xlsx"
HOLDOUT_PATH: Final[Path]   = INPUTS_DIR / "test_dataset.xlsx"
INVENTORY_PATH: Final[Path] = INPUTS_DIR / "inventory_transactions_clean.xlsx"
MODELS_DIR: Final[Path]     = Path("models"); MODELS_DIR.mkdir(exist_ok=True)

# --- Data Column Prefixes & Mappings ---
# Used to identify and parse different types of columns in the dataset.
CLS_PREFIX = "item_class_"
SKU_PREFIX = "product_name_"
ALL_SEGMENT_TYPES = ["Genre", "Slot", "Language", "Rating", "Duration"]
SEG_PREFIX  = {
    "genre_":   "Genre", "slot_":    "Slot", "language_":"Language",
    "rating_":  "Rating", "duration_category_":"Duration"
}

# --- Model & Rule Mining Hyperparameters ---
MIN_SUP, MIN_CONF, MIN_LIFT = 0.01, 0.05, 1.2 # Thresholds for association rules.
MIN_SEG_ROWS, MAX_REP       = 10, 8           # Min rows for a segment; max reps for basket weighting.

# --- Warning & Threading Control ---
# Suppress common, non-critical warnings from libraries for a cleaner user experience.
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning, module="mlxtend.frequent_patterns.association_rules")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Prevents thread over-subscription from underlying numerical libraries (e.g., NumPy/SciPy).
try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1, "blas")
except Exception:
    pass

# --- Recency Weighting Configuration ---
# Defines how to weight recent transactions more heavily in models.
RECENCY_WEIGHTING_CONFIG = {
    "apply": True,
    "timestamp_col": "timestamp",
    "tiers": [
        {"months_ago_end": 3, "weight_factor": 1.0},
        {"months_ago_end": 6, "weight_factor": 0.8},
        {"months_ago_end": 12, "weight_factor": 0.6},
        {"months_ago_end": 24, "weight_factor": 0.4},
        {"months_ago_end": float('inf'), "weight_factor": 0.2}
    ]
}

# ───────────────────────  DATACLASSES  ─────────────────────
# Using dataclasses for structured and type-safe parameter handling.

@dataclass
class RuleParams:
    """Parameters for Association Rule Mining (FP-Growth)."""
    min_support    : float = MIN_SUP
    min_confidence : float = MIN_CONF
    min_lift       : float = MIN_LIFT
    max_len_global : int   = 0

@dataclass
class RecParams:
    """Parameters for controlling the number of recommendations to generate."""
    top_classes  : int = 5
    skus_per_cls : int = 3

@dataclass
class EvaluationKValueParam:
    """Parameters for evaluation metrics, specifically the 'k' value."""
    k_for_rank_metrics : int = 10

# ───────────────────────  HELPER FUNCTIONS  ─────────────────────
# Small, reusable utility functions.

def cosine(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculates the cosine similarity between two vectors."""
    vec1_flat = np.asarray(vec1).flatten()
    vec2_flat = np.asarray(vec2).flatten()
    if vec1_flat.shape[0] != vec2_flat.shape[0] or vec1_flat.shape[0] == 0: return 0.0
    dot_product = np.dot(vec1_flat, vec2_flat)
    norm_vec1 = np.linalg.norm(vec1_flat)
    norm_vec2 = np.linalg.norm(vec2_flat)
    if norm_vec1 == 0.0 or norm_vec2 == 0.0: return 0.0
    return float(dot_product / (norm_vec1 * norm_vec2))

def ndcg_at_k(recs: list, actual: set, k_val: int) -> float:
    """Computes Normalized Discounted Cumulative Gain at k."""
    if not actual or k_val == 0: return 0.0
    eff_k = min(k_val, len(recs)); dcg = 0.0
    # Calculate Discounted Cumulative Gain
    for i in range(eff_k):
        if recs[i] in actual: dcg += 1.0 / np.log2(i + 2)
    # Calculate Ideal Discounted Cumulative Gain
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(eff_k, len(actual))))
    return dcg / idcg if idcg > 0 else 0.0

# ───────────────────────  DATA MANAGER  ─────────────────────
class DataManager:
    """
    A singleton-style class to handle loading, caching, and providing access
    to all datasets required by the dashboard.
    """
    def __init__(self):
        self._df_trx_raw_cache = None; self._df_hold_cache = None; self._df_inv_cache  = None
        self._sku2cls_cache = None; self._cls2sku_cache = None; self._masks_cache   = None

    @st.cache_data(show_spinner="📥 Loading raw transactions …")
    def _load_raw_trx(_self):
        """Loads and caches the main training transaction dataset."""
        df = pd.read_excel(TRAINING_PATH, parse_dates=[RECENCY_WEIGHTING_CONFIG["timestamp_col"]])
        if RECENCY_WEIGHTING_CONFIG["timestamp_col"] not in df.columns:
            raise ValueError(f"Training data must contain '{RECENCY_WEIGHTING_CONFIG['timestamp_col']}' column.")
        return df

    def get_weighted_trx(self, apply_recency_weighting: bool) -> pd.DataFrame:
        """
        Returns the transaction data, applying recency weighting if specified.
        The weighting adds a '__recency_weight' column to the DataFrame.
        """
        if self._df_trx_raw_cache is None: self._df_trx_raw_cache = self._load_raw_trx()
        df = self._df_trx_raw_cache.copy()
        if apply_recency_weighting and RECENCY_WEIGHTING_CONFIG.get("apply", False):
            ts_col = RECENCY_WEIGHTING_CONFIG["timestamp_col"]
            if ts_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[ts_col]) and not df[ts_col].isnull().all():
                max_date = df[ts_col].max()
                if pd.notna(max_date):
                    days_diff = (max_date - df[ts_col]).dt.days
                    df["age_in_months"] = (days_diff / 30.4375).fillna(0).astype(int)
                    df["__recency_weight"] = RECENCY_WEIGHTING_CONFIG["tiers"][-1]["weight_factor"]
                    for tier in sorted(RECENCY_WEIGHTING_CONFIG["tiers"], key=lambda x: x["months_ago_end"]):
                        df.loc[df["age_in_months"] < tier["months_ago_end"], "__recency_weight"] = tier["weight_factor"]
                else: df["__recency_weight"] = 1.0
            else: df["__recency_weight"] = 1.0
        else: df["__recency_weight"] = 1.0
        return df

    @st.cache_data(show_spinner="📥 Loading holdout data …")
    def _load_holdout_data(_self):
        """Loads and caches the holdout (test) dataset for evaluation."""
        if not HOLDOUT_PATH.exists():
            st.warning(f"Holdout data not found at {HOLDOUT_PATH}. Evaluation tab will be disabled.")
            return pd.DataFrame()
        df = pd.read_excel(HOLDOUT_PATH, parse_dates=[RECENCY_WEIGHTING_CONFIG["timestamp_col"]])
        if "__recency_weight" not in df.columns: df["__recency_weight"] = 1.0
        return df

    @st.cache_data(show_spinner="📥 Loading inventory …")
    def _load_inv(_self):
        """Loads and caches the inventory data for SKU-to-class mapping."""
        if not INVENTORY_PATH.exists():
            st.error(f"Inventory data not found at {INVENTORY_PATH}. Cannot proceed.")
            st.stop()
        return pd.read_excel(INVENTORY_PATH).drop_duplicates(subset="product_name", keep="first")

    @property
    def df_trx_raw(self) -> pd.DataFrame:
        """Provides access to the raw training data."""
        if self._df_trx_raw_cache is None: self._df_trx_raw_cache = self._load_raw_trx()
        return self._df_trx_raw_cache.copy()

    @property
    def df_holdout(self) -> pd.DataFrame:
        """Provides access to the holdout data."""
        if self._df_hold_cache is None: self._df_hold_cache = self._load_holdout_data()
        return self._df_hold_cache.copy()

    @property
    def sku2cls(self) -> dict:
        """Provides a cached dictionary mapping product names (SKUs) to item classes."""
        if self._sku2cls_cache is None:
            inv_df = self._df_inv_cache if self._df_inv_cache is not None else self._load_inv()
            self._df_inv_cache = inv_df
            self._sku2cls_cache = inv_df.set_index("product_name")["item_class"].str.upper().to_dict()
        return self._sku2cls_cache

    @property
    def cls2sku(self) -> dict:
        """Provides a cached dictionary mapping item classes to a list of their SKUs."""
        if self._cls2sku_cache is None:
            self._cls2sku_cache = {}
            raw_df_trx_for_map = self.df_trx_raw; current_sku2cls = self.sku2cls
            for col in raw_df_trx_for_map.filter(like=SKU_PREFIX).columns:
                sku = col[len(SKU_PREFIX):]; cls = current_sku2cls.get(sku)
                if cls: self._cls2sku_cache.setdefault(cls, []).append(col)
        return self._cls2sku_cache

    @property
    def masks(self) -> dict:
        """
        Provides a cached dictionary of pre-computed boolean masks for each filter
        segment (e.g., 'Genre · Action'), improving filtering performance.
        """
        if self._masks_cache is None:
            m = {}; raw_df_trx_for_masks = self.df_trx_raw
            for pref,lbl in SEG_PREFIX.items():
                for col in raw_df_trx_for_masks.columns:
                    if col.startswith(pref):
                        mask = raw_df_trx_for_masks[col].gt(0)
                        if mask.sum() >= MIN_SEG_ROWS:
                            val = col[len(pref):].replace('_',' ').title()
                            m[f"{lbl} · {val}"] = mask
            self._masks_cache = m
        return self._masks_cache
# Instantiate the data manager for global use in the app.
DM = DataManager()

# ───────────────────────  CORE LOGIC FUNCTIONS  ─────────────────────

def ctx_hash(ctx:dict, tag:str="", recency_applied_flag: bool = True) -> str:
    """
    Generates a unique hash for a given context and parameters, used for caching
    models and rule sets.
    """
    normalized_ctx = {k: sorted(v) if isinstance(v, list) else v for k, v in ctx.items()}
    raw_json_ctx = json.dumps(normalized_ctx, sort_keys=True); recency_config_tag_part = ""
    if recency_applied_flag and RECENCY_WEIGHTING_CONFIG.get("apply", False):
        recency_config_tag_part = hashlib.md5(json.dumps(RECENCY_WEIGHTING_CONFIG, sort_keys=True).encode()).hexdigest()[:6]
    full_string_to_hash = f"{CACHE_VER}{raw_json_ctx}{tag}{recency_config_tag_part}"
    return hashlib.md5(full_string_to_hash.encode()).hexdigest()[:10]

def subset_df(df:pd.DataFrame, ctx:dict) -> pd.DataFrame:
    """
    Filters a DataFrame based on the selected UI context (e.g., Genre, Slot).
    Uses the pre-computed masks for efficient filtering.
    """
    if not ctx: return df.copy()
    keep = pd.Series(True, index=df.index)
    for seg, vals_input in ctx.items():
        vals = vals_input if isinstance(vals_input, list) else [vals_input]
        or_mask = pd.Series(False, index=df.index); any_match_for_type = False
        for v_item in vals:
            k_candidate = f"{seg} · {str(v_item).title()}"
            m = DM.masks.get(k_candidate)
            if m is None:
                for mk_item in DM.masks:
                    if mk_item.startswith(f"{seg} ·") and str(v_item).lower() in mk_item.lower():
                        m = DM.masks[mk_item]; any_match_for_type = True; break
            if m is not None:
                aligned_mask = m.reindex(df.index, fill_value=False)
                or_mask |= aligned_mask; any_match_for_type = True
        if not any_match_for_type and vals: return pd.DataFrame(columns=df.columns)
        if any_match_for_type: keep &= or_mask
        else: return pd.DataFrame(columns=df.columns)
    if not keep.any(): return pd.DataFrame(columns=df.columns)
    return df.loc[keep].copy()

def row_rep_basket(df_ctx:pd.DataFrame, pref:str=CLS_PREFIX, cap:int=MAX_REP) -> pd.DataFrame:
    """
    Creates a "basket" representation of transactions for rule mining.
    It repeats rows based on the number of items and recency weight to give
    more importance to larger, more recent transactions.
    """
    qty_df = df_ctx.filter(like=pref).astype(int)
    if qty_df.empty: return pd.DataFrame(columns=qty_df.columns, dtype=bool)
    total_items_per_row = qty_df.sum(axis=1).to_numpy()
    rec_weight = df_ctx["__recency_weight"].to_numpy().astype(float) if "__recency_weight" in df_ctx.columns else np.ones_like(total_items_per_row, dtype=float)
    raw_reps = total_items_per_row * rec_weight
    capped_reps = np.minimum(np.round(raw_reps), cap).astype(int); capped_reps = np.maximum(capped_reps, 0)
    valid_idx_mask = capped_reps > 0
    if not np.any(valid_idx_mask): return pd.DataFrame(columns=qty_df.columns, dtype=bool)
    orig_indices = qty_df.index.to_numpy()[valid_idx_mask]; rep_counts  = capped_reps[valid_idx_mask]
    replicated_indices = np.repeat(orig_indices, rep_counts)
    if not replicated_indices.size: return pd.DataFrame(columns=qty_df.columns, dtype=bool)
    # Return a boolean DataFrame where True indicates presence in the basket.
    return qty_df.loc[replicated_indices].gt(0)

def compute_als_cosine(model: implicit.cpu.als.AlternatingLeastSquares, i: int, j: int) -> float:
    """Calculates cosine similarity between two item vectors from a trained ALS model."""
    if not (hasattr(model, 'item_factors') and model.item_factors is not None and
            0 <= i < model.item_factors.shape[0] and 0 <= j < model.item_factors.shape[0]): return 0.0
    v1, v2 = model.item_factors[i], model.item_factors[j]
    # Use pre-computed norms if available for performance
    if hasattr(model, 'item_norms') and model.item_norms is not None and \
       i < len(model.item_norms) and j < len(model.item_norms):
        n1, n2 = model.item_norms[i], model.item_norms[j]
        if n1 * n2 > 1e-9: return float(np.dot(v1, v2) / (n1 * n2 + 1e-12))
    # Fallback to computing norms on the fly
    nu, nv = np.linalg.norm(v1), np.linalg.norm(v2)
    if nu * nv < 1e-12: return 0.0
    return float(np.dot(v1, v2) / (nu * nv))

@st.cache_data(ttl=3600*12, show_spinner="⛏️ Mining association rules …")
def mine_rules(ctx:dict, p:RuleParams, recency_is_on: bool) -> pd.DataFrame:
    """Mines and caches association rules for a given context and parameters."""
    tag = f"RL{p.max_len_global}S{p.min_support:.4f}C{p.min_confidence:.4f}L{p.min_lift:.2f}"
    rules_hash = ctx_hash(ctx,tag=tag, recency_applied_flag=recency_is_on)
    pkl_suffix = "weighted" if recency_is_on and RECENCY_WEIGHTING_CONFIG.get("apply",False) else "unweighted"
    pkl = MODELS_DIR / f"rules_ctx_{pkl_suffix}_{rules_hash}.pkl"
    if pkl.exists(): return pickle.loads(pkl.read_bytes())
    df_rules = DM.get_weighted_trx(apply_recency_weighting=recency_is_on)
    df_ctx = subset_df(df_rules, ctx)
    empty_df = pd.DataFrame(columns=['antecedents','consequents','support','confidence','lift'])
    if df_ctx.empty: pkl.write_bytes(pickle.dumps(empty_df)); return empty_df
    basket = row_rep_basket(df_ctx, CLS_PREFIX, cap=MAX_REP)
    if basket.empty: pkl.write_bytes(pickle.dumps(empty_df)); return empty_df
    fp_kw = dict(min_support=p.min_support, use_colnames=True)
    if p.max_len_global > 0: fp_kw["max_len"] = p.max_len_global
    its = pd.DataFrame(); rules = empty_df.copy()
    try:
        its = fpgrowth(basket, **fp_kw)
        if not its.empty:
            temp_rules = association_rules(its, metric="lift", min_threshold=p.min_lift)
            if not temp_rules.empty: rules = temp_rules.query(f"confidence >= {p.min_confidence}").reset_index(drop=True)
    except Exception: pass
    pkl.write_bytes(pickle.dumps(rules)); return rules

def _convert_df_to_weighted_coo_st(ctx_df: pd.DataFrame, item_prefix: str, apply_recency: bool) -> tuple[sp.coo_matrix, list[str]]:
    """Converts a DataFrame slice into a sparse COO matrix for ALS model training."""
    item_cols = [c for c in ctx_df.columns if c.startswith(item_prefix)]
    if not item_cols: return sp.coo_matrix((ctx_df.shape[0], 0), dtype=np.float32), []
    mat = ctx_df[item_cols].to_numpy(np.float32, copy=False)
    weighted_mat = mat * ctx_df["__recency_weight"].to_numpy(np.float32)[:, np.newaxis] \
        if apply_recency and "__recency_weight" in ctx_df.columns and RECENCY_WEIGHTING_CONFIG.get("apply", False) else mat
    r, c = np.nonzero(weighted_mat)
    return sp.coo_matrix((weighted_mat[r, c], (r, c)), shape=(ctx_df.shape[0], len(item_cols)), dtype=np.float32), item_cols

@st.cache_resource(ttl=3600*12, show_spinner="⚙️ Training / loading ALS models …")
def load_als(ctx:dict, recency_is_on: bool):
    """Trains or loads cached ALS models for both item class and SKU levels."""
    als_hash = ctx_hash(ctx, tag="ALS_v3", recency_applied_flag=recency_is_on)
    pkl_suffix = "weighted" if recency_is_on and RECENCY_WEIGHTING_CONFIG.get("apply",False) else "unweighted"
    f_c = MODELS_DIR/f"als_cls_{pkl_suffix}_{als_hash}.npz"; f_s = MODELS_DIR/f"als_sku_{pkl_suffix}_{als_hash}.npz"
    meta_f = MODELS_DIR/f"als_meta_{pkl_suffix}_{als_hash}.pkl"
    if f_c.exists() and f_s.exists() and meta_f.exists():
        m_cls = implicit.cpu.als.AlternatingLeastSquares.load(f_c)
        m_sku = implicit.cpu.als.AlternatingLeastSquares.load(f_s)
        meta_d = pickle.loads(meta_f.read_bytes())
        return m_cls, m_sku, meta_d.get("cls", []), meta_d.get("sku", [])
    df_als = DM.get_weighted_trx(apply_recency_weighting=recency_is_on)
    df_ctx = subset_df(df_als, ctx)
    empty_als = implicit.als.AlternatingLeastSquares(factors=1, regularization=0.01, random_state=42, use_gpu=False)
    if df_ctx.empty:
        try: empty_als.save(f_c); empty_als.save(f_s)
        except: pass
        meta_f.write_bytes(pickle.dumps({"cls":[], "sku":[]})); return empty_als, empty_als, [], []

    def train_model(prefix, factors):
        coo, cols = _convert_df_to_weighted_coo_st(df_ctx, prefix, apply_recency=recency_is_on)
        model = implicit.als.AlternatingLeastSquares(factors=factors, iterations=20, regularization=0.1, random_state=42, use_gpu=False)
        if coo.shape[0] > 0 and coo.shape[1] > 0: model.fit(coo.T.tocsr())
        return model, cols
    m_cls, cls_cols = train_model(CLS_PREFIX, 60); m_sku, sku_cols = train_model(SKU_PREFIX, 50)
    try: m_cls.save(f_c); m_sku.save(f_s)
    except: pass
    meta_f.write_bytes(pickle.dumps({"cls":cls_cols, "sku":sku_cols})); return m_cls, m_sku, cls_cols, sku_cols

def partner_classes(target:str, ctx:dict, k:int, p:RuleParams, recency_on:bool):
    """
    Finds the top 'k' partner item classes for a given target class, combining
    association rule metrics (lift, confidence) with ALS similarity.
    """
    tgt_col = CLS_PREFIX + target.upper(); rules = mine_rules(ctx, p, recency_on)
    out_cols = ["partner_class", "als", "lift", "confidence", "support"]
    empty_df = pd.DataFrame(columns=out_cols)
    if rules.empty or not all(c in rules.columns for c in ['antecedents', 'consequents', 'support', 'confidence', 'lift']): return empty_df
    mask = (rules["antecedents"].apply(lambda s: tgt_col in s) | rules["consequents"].apply(lambda s: tgt_col in s))
    rs = rules.loc[mask, ['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
    if rs.empty: return empty_df
    def extract_partner(rw): sset = set(rw["antecedents"]) | set(rw["consequents"]); sset.discard(tgt_col); return next(iter(sset)).replace(CLS_PREFIX, "") if sset else None
    rs["partner_class"] = rs.apply(extract_partner, axis=1); rs.dropna(subset=["partner_class"], inplace=True)
    if rs.empty: return empty_df

    rs = rs.sort_values("lift", ascending=False).drop_duplicates("partner_class", keep="first")

    m_cls, _, cls_names, _ = load_als(ctx, recency_on); idx_map = {name: i for i, name in enumerate(cls_names)}
    rs["als"] = 0.0
    if tgt_col in idx_map and hasattr(m_cls, 'item_factors') and m_cls.item_factors is not None:
        t_idx = idx_map[tgt_col]
        if 0 <= t_idx < m_cls.item_factors.shape[0]:
            rs["als"] = rs["partner_class"].apply(lambda pc: compute_als_cosine(m_cls, t_idx, idx_map[CLS_PREFIX + pc.upper()]) if CLS_PREFIX + pc.upper() in idx_map and 0 <= idx_map[CLS_PREFIX + pc.upper()] < m_cls.item_factors.shape[0] else 0.0)

    # Sort by ALS similarity first, then by lift, and finally take the top k results.
    final_sorted_and_limited = rs.sort_values(["als", "lift"], ascending=[False, False]).head(k)

    for col_n in out_cols:
        if col_n not in final_sorted_and_limited.columns:
            final_sorted_and_limited[col_n] = np.nan if col_n != "partner_class" else "N/A_Error"

    return final_sorted_and_limited[out_cols].reset_index(drop=True)

def partner_skus(target: str, partner: str, ctx: dict, k: int, recency_on: bool) -> pd.DataFrame:
    """
    For a given target and partner class, recommends the top 'k' specific products (SKUs)
    from the partner class based on ALS similarity.
    """
    _, m_sku, _, sku_names = load_als(ctx, recency_on); idx_map = {name: i for i, name in enumerate(sku_names)}
    cls2sku_map = DM.cls2sku; target_up, partner_up = target.upper(), partner.upper()
    tgt_full = [c for c in cls2sku_map.get(target_up, []) if c in idx_map]
    prt_full = [c for c in cls2sku_map.get(partner_up, []) if c in idx_map]
    empty_out = pd.DataFrame(columns=["recommended_product_name", "product_als_similarity"])
    if not (tgt_full and prt_full and hasattr(m_sku, 'item_factors') and m_sku.item_factors is not None): return empty_out
    # Create an average vector representing the target class
    tgt_vecs = [m_sku.item_factors[idx_map[c]] for c in tgt_full if 0 <= idx_map[c] < m_sku.item_factors.shape[0]]
    if not tgt_vecs: return empty_out
    avg_vec = np.mean(tgt_vecs, axis=0); sims = []
    # Find similarity between the target vector and all products in the partner class
    for prt_sku_fullname in prt_full:
        prt_idx = idx_map[prt_sku_fullname]
        if 0 <= prt_idx < m_sku.item_factors.shape[0]:
            sim = cosine(avg_vec, m_sku.item_factors[prt_idx])
            sims.append((prt_sku_fullname[len(SKU_PREFIX):], sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    return pd.DataFrame(sims[:k], columns=["recommended_product_name", "product_als_similarity"])

def combo_table(ctx: dict, base: int, addon: int, k: int, recency_on:bool) -> pd.DataFrame:
    """
    Finds and formats popular 'combo deals' (e.g., buy 'base' items, get 'addon' items)
    based on association rules.
    """
    p = RuleParams(max_len_global=base + addon); rules = mine_rules(ctx, p, recency_on)
    empty_c = pd.DataFrame(columns=["base", "add", "support", "confidence", "lift"])
    if rules.empty or not all(c in rules.columns for c in ["antecedents", "consequents", "lift", "confidence", "support"]): return empty_c
    df = rules[(rules["antecedents"].apply(len) == base) & (rules["consequents"].apply(len) == addon) &
               (rules["lift"] >= p.min_lift) & (rules["confidence"] >= p.min_confidence)].copy()
    if df.empty: return empty_c
    fmt = lambda s: ", ".join(i.replace(CLS_PREFIX, "") for i in s)
    df["base"]  = df["antecedents"].apply(fmt); df["add"]   = df["consequents"].apply(fmt)
    return df.sort_values(["lift", "confidence"], ascending=[False, False]).head(k)[["base", "add", "support", "confidence", "lift"]].reset_index(drop=True)

def get_context_summary_stats(df_ctx_slice: pd.DataFrame, target_item_class: str, selected_ctx: dict):
    """Calculates summary statistics for the currently filtered data slice."""
    stats = {"num_total_transactions_in_slice": len(df_ctx_slice)}
    target_col = CLS_PREFIX + target_item_class.upper(); ts_col = RECENCY_WEIGHTING_CONFIG["timestamp_col"]
    if not df_ctx_slice.empty and target_col in df_ctx_slice.columns:
        stats["pct_with_target"] = (df_ctx_slice[target_col] > 0).mean() if stats["num_total_transactions_in_slice"] > 0 else 0.0
        stats["total_units_target"] = df_ctx_slice[target_col].sum()
        if ts_col in df_ctx_slice.columns and pd.api.types.is_datetime64_any_dtype(df_ctx_slice[ts_col]) and not df_ctx_slice[ts_col].isnull().all():
            min_d, max_d = df_ctx_slice[ts_col].min(), df_ctx_slice[ts_col].max()
            stats["date_range_min"] = min_d.strftime("%Y-%m-%d") if pd.notna(min_d) else "N/A"
            stats["date_range_max"] = max_d.strftime("%Y-%m-%d") if pd.notna(max_d) else "N/A"
        else: stats["date_range_min"], stats["date_range_max"] = "N/A", "N/A"
    else: stats.update({"pct_with_target":0.0, "total_units_target":0, "date_range_min":"N/A", "date_range_max":"N/A"})
    composition = {}
    if selected_ctx and not df_ctx_slice.empty:
        for seg_type, seg_values in selected_ctx.items():
            counts = {}
            for val in (seg_values if isinstance(seg_values, list) else [seg_values]):
                mk = f"{seg_type} · {str(val).title()}"
                if DM.masks and DM.df_trx_raw is not None and DM.masks.get(mk) is not None and DM.masks[mk].index.equals(DM.df_trx_raw.index):
                    counts[val] = df_ctx_slice.loc[DM.masks[mk].reindex(df_ctx_slice.index, fill_value=False)].shape[0]
            if counts: composition[seg_type] = counts
    stats["segment_composition"] = composition; return stats

def get_overall_kpis(data_mgr: DataManager, rule_p: RuleParams, rec_p: RecParams, eval_p: EvaluationKValueParam, target_item: str, recency_on: bool):
    """Calculates high-level Key Performance Indicators for the overview tab."""
    kpis = {"total_transactions_analyzed": len(data_mgr.df_trx_raw) if not data_mgr.df_trx_raw.empty else 0}

    global_rules_unweighted = mine_rules(ctx={}, p=dc_replace(rule_p, min_lift=0.1), recency_is_on=False)

    kpis["strongest_pairing_for_target_lift"] = 0.0
    kpis["strongest_pairing_for_target_partner"] = "N/A"

    if not global_rules_unweighted.empty and 'lift' in global_rules_unweighted.columns:
        target_col_name = CLS_PREFIX + target_item.upper()
        target_specific_rules = global_rules_unweighted[
            global_rules_unweighted['antecedents'].apply(lambda x: target_col_name in x) |
            global_rules_unweighted['consequents'].apply(lambda x: target_col_name in x)
        ].copy()

        if not target_specific_rules.empty:
            top_pairing_row_idx = target_specific_rules['lift'].idxmax()
            top_pairing_row = target_specific_rules.loc[top_pairing_row_idx]

            partner_items = set(top_pairing_row['antecedents']) | set(top_pairing_row['consequents'])
            partner_items.discard(target_col_name)

            partner_names_display = ", ".join([p.replace(CLS_PREFIX, "") for p in partner_items])

            kpis["strongest_pairing_for_target_lift"] = top_pairing_row['lift']
            kpis["strongest_pairing_for_target_partner"] = partner_names_display if partner_names_display else "N/A"

    eval_res = evaluate_recommendations_on_holdout_multi_metric(data_mgr, target_item, {}, rule_p, rec_p, eval_p, recency_on_for_generation=recency_on)
    kpis["overall_holdout_hit_rate"] = eval_res.get("hit_rate", np.nan)
    return kpis

def evaluate_recommendations_on_holdout_multi_metric(
    data_mgr: DataManager, target_item_class: str, ctx: dict, rule_p: RuleParams, rec_gen_p: RecParams,
    eval_k_p: EvaluationKValueParam, recency_on_for_generation:bool) -> dict:
    """
    Evaluates generated recommendations against the holdout set and calculates
    various ranking and performance metrics.
    """
    partner_df, sku_map = generate_recommendations(data_mgr, target_item_class, ctx, rule_p, rec_gen_p, recency_on_for_generation)
    recs_ordered = []; seen_skus = set()
    if sku_map:
        order = partner_df["partner_item_class"].tolist() if not partner_df.empty and "partner_item_class" in partner_df.columns else list(sku_map.keys())
        for p_name in order:
            products_df = sku_map.get(p_name, pd.DataFrame())
            if "recommended_product_name" in products_df.columns and not products_df.empty:
                for product_name in products_df["recommended_product_name"].tolist():
                    if product_name not in seen_skus: recs_ordered.append(product_name); seen_skus.add(product_name)
    k_eval = min(eval_k_p.k_for_rank_metrics, len(recs_ordered)) if recs_ordered else 0
    def_metrics = {"hit_rate": 0.0, "precision_at_k": 0.0, "recall_at_k": 0.0, "ndcg_at_k": 0.0, "evaluated_sessions": 0, "k_value": k_eval}
    if not recs_ordered: return {**def_metrics, "error": "No products recommended."}
    hold_ctx = subset_df(data_mgr.df_holdout, ctx)
    if hold_ctx.empty: return {**def_metrics, "error": "Empty holdout slice for context."}
    tgt_col = CLS_PREFIX + target_item_class.upper()
    if tgt_col not in hold_ctx.columns: return {**def_metrics, "error": f"Target '{tgt_col}' missing in holdout."}
    sessions_tgt = hold_ctx[hold_ctx[tgt_col] > 0]
    if sessions_tgt.empty: return {**def_metrics, "error": "No relevant sessions in holdout."}
    num_eval, hits, prec_s, rec_s, ndcg_s = 0, 0, 0.0, 0.0, 0.0
    for _, row in sessions_tgt.iterrows():
        purchased = {c[len(SKU_PREFIX):] for c, q in row.items() if isinstance(c, str) and c.startswith(SKU_PREFIX) and q > 0}
        if not purchased: continue
        num_eval += 1; hits += 1 if set(recs_ordered).intersection(purchased) else 0
        if k_eval > 0:
            top_k = recs_ordered[:k_eval]; rel_in_topk = set(top_k).intersection(purchased)
            prec_s += len(rel_in_topk) / k_eval; rec_s += len(rel_in_topk) / len(purchased)
            ndcg_s += ndcg_at_k(top_k, purchased, k_eval)
    results = {"evaluated_sessions": num_eval, "k_value": k_eval}
    if num_eval > 0:
        results.update({"hit_rate": hits / num_eval, "precision_at_k": (prec_s / num_eval) if k_eval > 0 else 0.0,
                        "recall_at_k": (rec_s / num_eval) if k_eval > 0 else 0.0, "ndcg_at_k": (ndcg_s / num_eval) if k_eval > 0 else 0.0})
    else: results.update({k: np.nan for k in ["hit_rate","precision_at_k","recall_at_k","ndcg_at_k"]}); results = {**def_metrics, **results}
    return results

def generate_recommendations(data_mgr: DataManager, target_item_class: str, ctx: dict | None, rule_p: RuleParams, rec_gen_p: RecParams, recency_on: bool):
    """
    Orchestrates the two-stage recommendation process:
    1. Find partner item classes.
    2. For each partner class, find specific product SKUs.
    """
    ctx_eff = ctx if ctx is not None else {}
    pc_df_from_partner_classes = partner_classes(target_item_class, ctx_eff, rec_gen_p.top_classes, rule_p, recency_on)
    sku_map = {}
    if not pc_df_from_partner_classes.empty:
        for _, row in pc_df_from_partner_classes.iterrows():
            p_name = row["partner_class"]
            if pd.notna(p_name) and p_name != "N/A_Error":
                 product_df = partner_skus(target_item_class, p_name, ctx_eff, rec_gen_p.skus_per_cls, recency_on)
                 sku_map[p_name] = product_df
    pc_df_final = pc_df_from_partner_classes.rename(columns={
        "partner_class":"partner_item_class",
        "als":"class_als_similarity"
        }, errors='ignore')
    return pc_df_final, sku_map

def format_context_for_display(ctx_dict: dict) -> str:
    """Creates a human-readable string from the context dictionary for display."""
    if not ctx_dict: return "Global (No Filters Applied)"
    parts = []
    for key, values in ctx_dict.items():
        value_str = ", ".join(map(str, values)) if isinstance(values, list) else str(values)
        parts.append(f"{key}: {value_str}")
    return "Filters: " + " | ".join(parts)

# ───────────────────────────  STREAMLIT UI  ───────────────────────────

st.title("🎬 Village Cinemas: F&B Recommender App")
st.subheader("Discover Popular Food & Beverage Pairings and Recommendation Performance")

with st.sidebar:
    st.title("⚙️ Dashboard Controls")
    # Trigger data loading by accessing a property. Handles errors if files are missing.
    if DM.df_trx_raw.empty:
        st.error("Training data could not be loaded. Please ensure train_dataset.xlsx is in the input_datasets folder.")
        st.stop()
        
    all_cls_options = sorted({c.replace(CLS_PREFIX,"") for c in DM.df_trx_raw.columns if c.startswith(CLS_PREFIX)})
    default_idx = all_cls_options.index("POPCORN") if "POPCORN" in all_cls_options else 0
    target_item_select = st.selectbox("🎯 Select Target Item", all_cls_options, index=default_idx, key="target_select_main_sb_v14")

    st.markdown("---")
    st.subheader("📊 Filter Recommendations By:")
    ctx_ui_input = {}
    # Access masks to ensure they are computed before use.
    _ = DM.masks

    filter_widget_keys = []
    for seg_type_filter in ALL_SEGMENT_TYPES:
        options = sorted({m.split(" · ")[1] for m in (DM.masks or {}) if m.startswith(f"{seg_type_filter} ·")})
        if options:
            widget_key = f"ms_{seg_type_filter.lower()}_main_sb_v14"
            filter_widget_keys.append(widget_key)
            if widget_key not in st.session_state:
                st.session_state[widget_key] = []
            sel = st.multiselect(f"{seg_type_filter}", options, key=widget_key)
            if sel: ctx_ui_input[seg_type_filter] = sel

    def reset_filters_callback():
        """Callback function for the reset button to clear filter selections."""
        for key_to_reset in filter_widget_keys:
            if key_to_reset in st.session_state:
                st.session_state[key_to_reset] = []

    st.button("🔄 Reset All Filters", on_click=reset_filters_callback, key="reset_filters_btn_v14")

    st.markdown("---")

    RECENCY_SB_KEY = 'recency_toggle_val_sb_v14'
    if RECENCY_SB_KEY not in st.session_state:
        st.session_state[RECENCY_SB_KEY] = RECENCY_WEIGHTING_CONFIG.get("apply", True)
    actual_recency_applied_ui = st.checkbox(
        "📅 Prioritize Recent Purchases?",
        key=RECENCY_SB_KEY,
        help="If checked, newer sales data has more weight in finding recommendations."
    )

    EVAL_SB_KEY = 'eval_toggle_val_sb_v14'
    if EVAL_SB_KEY not in st.session_state:
        st.session_state[EVAL_SB_KEY] = True
    show_eval_tab_ui = st.checkbox(
        "📈 Show Recommendation Evaluation Tab?",
        key=EVAL_SB_KEY
    )

    st.markdown("---")
    st.subheader("⚙️ Recommendation Display Options")
    num_top_partners_slider = st.slider("Partner Categories to Show", 1, 10, 5, key="num_partners_main_sb_v14")
    num_skus_per_partner_slider = st.slider("Products per Partner Category", 1, 6, 3, key="num_skus_main_sb_v14")
    num_combos_slider = st.slider("Combo Examples to Show", 1, 10, 5, key="num_combos_main_sb_v14")

    st.markdown("---")
    st.subheader("📘 Understanding the Scores")

    with st.expander("What is ALS Similarity?"):
        st.caption("""
        **ALS Similarity (ALS Sim.):** This score (from 0 to 1) comes from a machine learning model.
        It finds items often bought by similar customers, even in different purchases.
        A higher score means items are more alike based on buying patterns.
        """)
    with st.expander("What is Lift?"):
        st.caption("""
        **Lift:** Shows how much more likely customers buy two items together versus by chance.
        A Lift over 1 means they're a good pair. E.g., Lift of 2 means twice as likely.
        """)
    with st.expander("What is Confidence?"):
        st.caption("""
        **Confidence:** If a customer buys Item A, what's the chance they also buy Item B? That's Confidence.
        E.g., 60% Confidence (Popcorn -> Drink) means 60% of Popcorn buyers also got a Drink.
        """)
    with st.expander("What is Support?"):
        st.caption("""
        **Support:** How often an item (or items) appears in all sales. High support = common; Low support = rare.
        """)
    with st.expander("What is Hit Rate@k?"):
        st.caption("""
        **Hit Rate@k:** When testing our top 'k' recommendations, what percentage of times did we correctly suggest at least one item the customer actually bought?
        """)
    with st.expander("What is Precision@k?"):
        st.caption("""
        **Precision@k:** Out of our top 'k' recommendations, what percentage were items the customer actually bought? High precision means fewer irrelevant suggestions.
        """)
    with st.expander("What is NDCG@k?"):
        st.caption("""
        **NDCG@k:** A score (0 to 1) measuring if we ranked the best recommendations at the top. Higher is better.
        """)

    st.markdown("---")
    with st.expander("ℹ️ About This Dashboard", expanded=False):
        st.markdown(f"""
This dashboard helps identify complementary food and beverage pairings using sales data.
*   **Methodology:** Recommendations are based on Association Rule Mining (FP-Growth) and Collaborative Filtering (ALS).
*   **Recency Weighting:** If 'Prioritize Recent Purchases?' is active, newer transactions have more influence.
""")

# Prepare parameters based on UI controls for the main app logic
current_rule_params = RuleParams()
current_rec_params = RecParams(top_classes=num_top_partners_slider, skus_per_cls=num_skus_per_partner_slider)
k_for_eval_metrics = max(1, num_top_partners_slider * num_skus_per_partner_slider)
current_eval_k_params = EvaluationKValueParam(k_for_rank_metrics=k_for_eval_metrics)
overall_k_params = EvaluationKValueParam(k_for_rank_metrics=10)

# Main panel display
st.header(f"Recommendations for {target_item_select}")
if not ctx_ui_input:
    st.info("💡 Showing recommendations based on all data. Use sidebar controls to apply filters for a more specific view.")
else:
    st.caption(f"{format_context_for_display(ctx_ui_input)}")

st.caption(f"Prioritizing Recent Purchases: {'✅ Active' if st.session_state[RECENCY_SB_KEY] and RECENCY_WEIGHTING_CONFIG.get('apply', False) else '❌ Inactive'}")

tab_titles = ["📊 Overview", "🤝 Item Pairings & Products", "🎁 Combo Examples"]
if st.session_state.get(EVAL_SB_KEY, True):
    tab_titles.append("📈 Recommendation Evaluation")
tab_titles.append("🔀 Scenario Explorer")
tabs = st.tabs(tab_titles)

# Define tab indices for clarity
tab_overview_idx, tab_partners_skus_idx, tab_combos_idx = 0, 1, 2
eval_tab_offset = 1 if st.session_state.get(EVAL_SB_KEY, True) else 0

# --- Tab 1: Overview ---
with tabs[tab_overview_idx]:
    st.subheader("🔍 Key Insights & Data Snapshot")
    kpis = get_overall_kpis(DM, current_rule_params, current_rec_params, overall_k_params, target_item_select, st.session_state[RECENCY_SB_KEY])

    cols = st.columns(3)
    cols[0].metric("Total Sales Records Analyzed", f"{kpis.get('total_transactions_analyzed',0):,}")

    strongest_partner = kpis.get("strongest_pairing_for_target_partner", "N/A")
    strongest_lift = kpis.get("strongest_pairing_for_target_lift", 0.0)

    metric_label = f"Strongest Pairing for {target_item_select} (Lift)"
    if strongest_partner not in ["N/A", "Lift data missing"] and strongest_partner:
        metric_help = f"Paired with: {strongest_partner}"
    else:
        metric_help = "No specific strong pairing found for this target with current global rule settings."
        if strongest_partner == "Lift data missing":
            metric_help = "Could not determine strongest pairing (data issue)."
    cols[1].metric(metric_label, f"{strongest_lift:.2f}", help=metric_help)

    hr_val = kpis.get('overall_holdout_hit_rate', np.nan)
    cols[2].metric(f"Overall Hit Rate@{overall_k_params.k_for_rank_metrics} (Target: {target_item_select})", f"{hr_val:.1%}" if pd.notna(hr_val) else "N/A",
                   help="Based on test data. How often top recommendations included an item actually bought with the target.")

    with st.expander("📋 Snapshot of Filtered Data", expanded=True):
        df_summary = DM.get_weighted_trx(apply_recency_weighting=st.session_state[RECENCY_SB_KEY])
        df_ctx_summary = subset_df(df_summary, ctx_ui_input)
        stats_summary = get_context_summary_stats(df_ctx_summary, target_item_select, ctx_ui_input)
        cols_snap = st.columns([2, 2, 2, 3])
        cols_snap[0].metric("Transactions in View", f"{stats_summary.get('num_total_transactions_in_slice',0):,}")
        cols_snap[1].metric(f"% With {target_item_select}", f"{stats_summary.get('pct_with_target',0.0):.1%}")
        cols_snap[2].metric(f"Units of {target_item_select}", f"{stats_summary.get('total_units_target',0):,}")
        cols_snap[3].metric("Date Range in View", f"{stats_summary.get('date_range_min','N/A')} to {stats_summary.get('date_range_max','N/A')}")

        if stats_summary.get("segment_composition"):
            st.write("**Breakdown by Active Filters:**")
            for seg_type, counts in stats_summary["segment_composition"].items():
                if counts:
                    df_comp = pd.DataFrame(list(counts.items()), columns=['Value', 'Transactions'])
                    chart = alt.Chart(df_comp).mark_bar().encode(x='Transactions:Q', y=alt.Y('Value:N', sort='-x', title=seg_type), tooltip=['Value', 'Transactions']).properties(title=f"{seg_type} Distribution")
                    st.altair_chart(chart, use_container_width=True)

    st.subheader(f"Sales Trend for '{target_item_select}' (with active filters)")
    target_plot_col = CLS_PREFIX + target_item_select.upper()
    if not df_ctx_summary.empty and target_plot_col in df_ctx_summary.columns:
        df_target_plot = df_ctx_summary[df_ctx_summary[target_plot_col] > 0].copy()
        ts_plot_col = RECENCY_WEIGHTING_CONFIG["timestamp_col"]
        if not df_target_plot.empty and ts_plot_col in df_target_plot.columns and pd.api.types.is_datetime64_any_dtype(df_target_plot[ts_plot_col]):
            df_target_plot[ts_plot_col] = pd.to_datetime(df_target_plot[ts_plot_col])
            df_target_plot["month_year"] = df_target_plot[ts_plot_col].dt.to_period("M").astype(str)
            monthly_counts = df_target_plot.groupby("month_year").size().reset_index(name="sales_count")
            if not monthly_counts.empty:
                line = alt.Chart(monthly_counts).mark_line(point=True).encode(x=alt.X("month_year:O", title="Month", sort=None), y=alt.Y("sales_count:Q", title="Number of Sales"), tooltip=["month_year", "sales_count"]).properties(title=f"Monthly Sales of {target_item_select}").interactive()
                st.altair_chart(line, use_container_width=True)
            else: st.info(f"No monthly sales data for '{target_item_select}' with current filters.")
        else: st.info(f"No sales of '{target_item_select}' or valid timestamp for trend plot with current filters.")
    else: st.info(f"Target '{target_item_select}' not found or data empty for trend plot.")

# --- Tab 2: Item Pairings & Products ---
with tabs[tab_partners_skus_idx]:
    st.header("🤝 Recommended Item Pairings & Specific Products")
    partners_df_original, skus_map = generate_recommendations(DM, target_item_select, ctx_ui_input, current_rule_params, current_rec_params, st.session_state[RECENCY_SB_KEY])
    partners_df = partners_df_original.copy()

    sort_options_map = {
        "ALS Sim (Default)": (["class_als_similarity", "lift"], [False, False]),
        "Lift": (["lift", "class_als_similarity"], [False, False]),
        "Confidence": (["confidence", "lift"], [False, False]),
        "Support": (["support", "lift"], [False, False])
    }
    sort_by_selection = st.selectbox("Sort partner categories by:", options=list(sort_options_map.keys()), key="partner_sort_select_v14")
    sort_columns_internal, sort_ascending_internal = sort_options_map[sort_by_selection]

    if not partners_df.empty:
        valid_sort_cols = [col for col in sort_columns_internal if col in partners_df.columns]
        if valid_sort_cols:
            valid_sort_ascending = [asc for col, asc in zip(sort_columns_internal, sort_ascending_internal) if col in valid_sort_cols]
            if valid_sort_ascending:
                 partners_df = partners_df.sort_values(by=valid_sort_cols, ascending=valid_sort_ascending).reset_index(drop=True)

    if not partners_df.empty:
        st.subheader("✨ Top Insights")
        top_p = partners_df.iloc[0]
        st.markdown(f"*   **Strongest Partner Category**: For **{target_item_select}**, **{top_p['partner_item_class']}** shows high association (ALS Sim: {top_p['class_als_similarity']:.3f}, Lift: {top_p['lift']:.2f}).")

        if top_p['partner_item_class'] in skus_map and not skus_map[top_p['partner_item_class']].empty:
            top_s = skus_map[top_p['partner_item_class']].iloc[0]
            st.markdown(f"*   **Top Product with {top_p['partner_item_class']}**: **{top_s['recommended_product_name']}** (Product ALS Sim: {top_s['product_als_similarity']:.3f}).")
        else: st.markdown(f"*   No specific product suggestions stood out for {top_p['partner_item_class']}.")

        st.markdown("---")
        st.subheader("🏆 Top Partner Categories to Consider")
        cols_display = st.columns([3,2])
        with cols_display[0]:
            partners_display_df = partners_df.rename(columns={"partner_item_class":"Partner Category", "class_als_similarity":"ALS Sim", "lift":"Lift", "confidence":"Confidence", "support":"Support"})
            ordered_cols = ["Partner Category", "ALS Sim", "Lift", "Confidence", "Support"]
            display_cols = [col for col in ordered_cols if col in partners_display_df.columns]
            st.dataframe(partners_display_df[display_cols].style.background_gradient(subset=["ALS Sim"], cmap="Blues")
                         .background_gradient(subset=["Lift"], cmap="Greens")
                         .format({"ALS Sim":"{:.4f}", "Lift":"{:.2f}", "Confidence":"{:.1%}", "Support":"{:.2%}"}),
                         use_container_width=True, hide_index=True)
            st.download_button("📥 Download Partner Categories", partners_df.to_csv(index=False).encode('utf-8'), f'partners_{target_item_select}.csv', 'text/csv', key="dl_partners_tab_v14")
        with cols_display[1]:
            chart_df_altair = partners_df.rename(columns={"class_als_similarity":"ALS Sim", "partner_item_class": "Partner"})
            scatter = alt.Chart(chart_df_altair).mark_circle(size=100).encode(
                x=alt.X("lift:Q", title="Lift"), y=alt.Y("ALS Sim:Q", title="Category ALS Sim"),
                color=alt.Color("Partner:N", legend=None), tooltip=["Partner", "ALS Sim", "lift", "confidence", "support"]
            ).properties(title="Partner Category: Lift vs. ALS Similarity").interactive()
            st.altair_chart(scatter, use_container_width=True)

        st.markdown("---")
        st.subheader("🎯 Specific Product Suggestions from Partner Categories")
        for idx, p_row_tuple in enumerate(partners_df.iterrows()):
            _, p_row = p_row_tuple
            p_name = p_row["partner_item_class"]
            exp_title = f"Product suggestions for '{p_name}' (ALS Sim: {p_row['class_als_similarity']:.3f}, Lift: {p_row['lift']:.2f})"
            with st.expander(exp_title):
                current_products = skus_map.get(p_name, pd.DataFrame())
                if current_products.empty: st.write("*No specific products stood out for this category.*")
                else:
                    products_disp = current_products.rename(columns={"recommended_product_name":"Product Name", "product_als_similarity":"Product ALS Sim"})
                    bar_chart = alt.Chart(products_disp).mark_bar().encode(x=alt.X("Product ALS Sim:Q"), y=alt.Y("Product Name:N", sort="-x"), tooltip=["Product Name", alt.Tooltip("Product ALS Sim:Q", format=".4f")])
                    cols_sku_exp = st.columns([2,3])
                    with cols_sku_exp[0]:
                        st.dataframe(products_disp.style.bar(subset=["Product ALS Sim"], color="#1f77b4", align="zero").format({"Product ALS Sim":"{:.4f}"}), hide_index=True, use_container_width=True)
                        st.download_button(f"📥 Products for {p_name}", current_products.to_csv(index=False).encode('utf-8'), f'products_{p_name}.csv', 'text/csv', key=f"dl_product_tab_v14_{idx}")
                    with cols_sku_exp[1]: st.altair_chart(bar_chart, use_container_width=True)
    else: st.info(f"No significant item pairings found for **{target_item_select}** with the current filters.")

# --- Tab 3: Combo Examples ---
with tabs[tab_combos_idx]:
    st.header("🎁 Example Combo Deals (Based on Item Category Co-occurrence)")
    cols_combo = st.columns(2)
    with cols_combo[0]:
        st.subheader("If Customer Buys 2 Items, Offer 1 More")
        c21_df = combo_table(ctx_ui_input, 2, 1, num_combos_slider, recency_on=st.session_state[RECENCY_SB_KEY])
        if c21_df.empty: st.info("No 'Buy 2 Get 1' combo examples found with current settings.")
        else:
            st.dataframe(c21_df.style.format({"support":"{:.2%}", "confidence":"{:.1%}", "lift":"{:.2f}"}), hide_index=True)
            st.download_button("📥 Download B2G1 Combos", c21_df.to_csv(index=False).encode('utf-8'), 'b2g1_combos.csv', 'text/csv', key="dl_b2g1_tab_v14")
    with cols_combo[1]:
        st.subheader("If Customer Buys 1 Item, Offer 1 More")
        c11_df = combo_table(ctx_ui_input, 1, 1, num_combos_slider, recency_on=st.session_state[RECENCY_SB_KEY])
        if c11_df.empty: st.info("No 'Buy 1 Get 1' combo examples found with current settings.")
        else:
            st.dataframe(c11_df.style.format({"support":"{:.2%}", "confidence":"{:.1%}", "lift":"{:.2f}"}), hide_index=True)
            st.download_button("📥 Download B1G1 Combos", c11_df.to_csv(index=False).encode('utf-8'), 'b1g1_pairs.csv', 'text/csv', key="dl_b1g1_tab_v14")

# --- Tab 4: Recommendation Evaluation (Conditional) ---
if st.session_state.get(EVAL_SB_KEY, True):
    with tabs[tab_combos_idx + eval_tab_offset]:
        st.header("📈 Evaluating How Well Product Recommendations Perform")
        if DM.df_holdout.empty:
            st.warning("Holdout data could not be loaded. Cannot perform evaluation.")
        else:
            st.markdown(f"Testing recommendations for **{target_item_select}** (with {format_context_for_display(ctx_ui_input).lower()}) against a separate test dataset.")
            st.caption(f"Metrics below are for the top **{k_for_eval_metrics}** products suggested by the system.")

            with st.spinner("⚙️ Calculating evaluation scores... Please wait."):
                eval_data = evaluate_recommendations_on_holdout_multi_metric(
                    DM, target_item_select, ctx_ui_input,
                    current_rule_params, current_rec_params, current_eval_k_params,
                    recency_on_for_generation=st.session_state[RECENCY_SB_KEY])

            if "error" in eval_data: st.warning(f"Could not calculate metrics: {eval_data['error']}")
            else:
                k_res = eval_data.get('k_value', k_for_eval_metrics)
                cols_eval = st.columns(4)
                cols_eval[0].metric("Test Purchases Evaluated", f"{eval_data.get('evaluated_sessions',0):,}")
                hr, pr, rc, ndcg_val = eval_data.get('hit_rate',np.nan), eval_data.get('precision_at_k',np.nan), eval_data.get('recall_at_k',np.nan), eval_data.get('ndcg_at_k',np.nan)
                cols_eval[1].metric(f"Hit Rate@{k_res}", f"{hr:.1%}" if pd.notna(hr) else "N/A")
                cols_eval[2].metric(f"Precision@{k_res}", f"{pr:.1%}" if pd.notna(pr) else "N/A")
                cols_eval[3].metric(f"NDCG@{k_res}", f"{ndcg_val:.3f}" if pd.notna(ndcg_val) else "N/A")

                st.markdown("---")
                st.subheader("Performance Metrics Chart")
                metrics_df_eval = pd.DataFrame({
                    "Metric": [f"Hit Rate@{k_res}", f"Precision@{k_res}", f"Recall@{k_res}", f"NDCG@{k_res}"],
                    "Score": [hr, pr, rc, ndcg_val]}).dropna(subset=['Score'])

                if not metrics_df_eval.empty:
                    chart_eval = alt.Chart(metrics_df_eval).mark_bar(cornerRadiusEnd=3).encode(
                        x=alt.X("Score:Q", title="Metric Score", scale=alt.Scale(domain=[0,1])),
                        y=alt.Y("Metric:N", sort=None, title=None, axis=alt.Axis(labelFontSize=12)),
                        color=alt.Color("Metric:N", legend=None, scale=alt.Scale(scheme='category10')),
                        tooltip=[alt.Tooltip("Metric:N"), alt.Tooltip("Score:Q", format=",.3%")]
                    ).properties(height=250, title=alt.TitleParams(text="Recommendation Quality Scores", anchor='middle', fontSize=16))
                    st.altair_chart(chart_eval, use_container_width=True)
                else: st.info("No evaluation scores available to display.")

# --- Tab 5: Scenario Explorer ---
what_if_tab_index = tab_combos_idx + eval_tab_offset + 1
with tabs[what_if_tab_index]:
    st.header("🔀 Explore 'What If' Scenarios")
    st.write(f"Compare recommendations for **{target_item_select}** by changing filters for two different scenarios.")
    st.caption(f"Prioritizing Recent Purchases: {'✅ Active' if st.session_state[RECENCY_SB_KEY] and RECENCY_WEIGHTING_CONFIG.get('apply', False) else '❌ Inactive'}")

    cols_whatif_main = st.columns(2)
    whatif_rules_params = RuleParams(); whatif_recs_params = RecParams(top_classes=3, skus_per_cls=2)

    scenario_results_A = {}
    scenario_results_B = {}

    def display_what_if_scenario(column, label:str, results_dict: dict):
        """Renders the UI and logic for a single scenario column in the explorer."""
        with column:
            st.subheader(f"Scenario {label}")
            ctx_scen = {}
            for seg_type_wf in ALL_SEGMENT_TYPES:
                opts_wf = sorted({m.split(" · ")[1] for m in (DM.masks or {}) if m.startswith(f"{seg_type_wf} ·")})
                if opts_wf:
                    sel_wf = st.multiselect(f"{seg_type_wf} ({label})", opts_wf, key=f"ms_whatif_{seg_type_wf.lower()}_{label.lower()}_v14")
                    if sel_wf: ctx_scen[seg_type_wf] = sel_wf

            st.markdown(f"**{format_context_for_display(ctx_scen)}**")

            with st.spinner(f"Generating recommendations for Scenario {label}..."):
                pc_df_wf, sku_map_wf = generate_recommendations(DM, target_item_select, ctx_scen, whatif_rules_params, whatif_recs_params, st.session_state[RECENCY_SB_KEY])

            results_dict["partners"] = pc_df_wf
            results_dict["skus"] = sku_map_wf

            st.write(f"Top Partner Categories ({label}):")
            if not pc_df_wf.empty:
                pc_disp_wf = pc_df_wf.rename(columns={"partner_item_class":"Partner", "class_als_similarity":"ALS Sim", "lift":"Lift", "confidence":"Conf.", "support":"Supp."})
                if "Partner" not in pc_disp_wf.columns and "partner_class" in pc_disp_wf.columns:
                     pc_disp_wf = pc_disp_wf.rename(columns={"partner_class":"Partner"})
                display_cols_wf = ["Partner", "ALS Sim", "Lift", "Conf.", "Supp."]
                existing_display_cols_wf = [col for col in display_cols_wf if col in pc_disp_wf.columns]
                st.dataframe(pc_disp_wf[existing_display_cols_wf].style.format(precision=2), hide_index=True, height=150)
            else: st.caption(f"No partner categories found for Scenario {label}.")

            if sku_map_wf:
                 st.write(f"Product Suggestions ({label}):")
                 for idx_whatif_exp, (p_name_wf, s_df_wf) in enumerate(sku_map_wf.items()):
                    if not s_df_wf.empty:
                        with st.expander(f"Products for '{p_name_wf}' ({label})"):
                            s_disp_wf = s_df_wf.rename(columns={"recommended_product_name":"Product", "product_als_similarity":"ALS Sim"})
                            st.dataframe(s_disp_wf.style.format({"ALS Sim":"{:.3f}"}), hide_index=True)

    display_what_if_scenario(cols_whatif_main[0], "A", scenario_results_A)
    display_what_if_scenario(cols_whatif_main[1], "B", scenario_results_B)

    if scenario_results_A and scenario_results_B:
        st.markdown("---")
        st.subheader("🔎 Scenario Comparison Highlights")

        pc_A = scenario_results_A.get("partners", pd.DataFrame())
        pc_B = scenario_results_B.get("partners", pd.DataFrame())
        skus_A_map = scenario_results_A.get("skus", {})
        skus_B_map = scenario_results_B.get("skus", {})

        partners_A_set = set(pc_A["partner_item_class"]) if not pc_A.empty and "partner_item_class" in pc_A.columns else set()
        partners_B_set = set(pc_B["partner_item_class"]) if not pc_B.empty and "partner_item_class" in pc_B.columns else set()

        unique_to_A_partners = partners_A_set - partners_B_set
        unique_to_B_partners = partners_B_set - partners_A_set
        common_partners = partners_A_set.intersection(partners_B_set)

        if not unique_to_A_partners and not unique_to_B_partners and not common_partners:
            st.info("No partner categories were recommended in either scenario to compare.")
        else:
            if unique_to_A_partners:
                st.markdown(f"**Partner Categories only in Scenario A:** {', '.join(sorted(list(unique_to_A_partners)))}")
            if unique_to_B_partners:
                st.markdown(f"**Partner Categories only in Scenario B:** {', '.join(sorted(list(unique_to_B_partners)))}")
            if not common_partners and (unique_to_A_partners or unique_to_B_partners):
                st.info("No common partner categories found between scenarios to compare specific products.")

        if common_partners:
            st.markdown("**Differences in Product Suggestions for Common Partner Categories:**")
            found_sku_diff = False
            for partner in sorted(list(common_partners)):
                products_A_df = skus_A_map.get(partner, pd.DataFrame())
                products_B_df = skus_B_map.get(partner, pd.DataFrame())

                products_A_set = set(products_A_df["recommended_product_name"]) if not products_A_df.empty and "recommended_product_name" in products_A_df else set()
                products_B_set = set(products_B_df["recommended_product_name"]) if not products_B_df.empty and "recommended_product_name" in products_B_df else set()

                unique_products_A = products_A_set - products_B_set
                unique_products_B = products_B_set - products_A_set

                if unique_products_A or unique_products_B:
                    found_sku_diff = True
                    with st.expander(f"Product differences for '{partner}'"):
                        if unique_products_A:
                            st.markdown(f"*   **Only in Scenario A:** {', '.join(sorted(list(unique_products_A)))}")
                        if unique_products_B:
                            st.markdown(f"*   **Only in Scenario B:** {', '.join(sorted(list(unique_products_B)))}")
            if not found_sku_diff and common_partners: # Added check for common_partners
                 st.info("For common partner categories, the product suggestions are the same in both scenarios.")