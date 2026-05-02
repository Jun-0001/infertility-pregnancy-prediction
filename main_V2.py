"""
main.py — IVF 난임 예측 파이프라인 (6-Model Ensemble)

폴더 구조:
    project/
    ├── data/
    │   ├── train.csv / test.csv / sample_submission.csv
    ├── pipeline/
    │   ├── schema.py
    │   ├── preprocess.py
    │   └── feature_engineering.py   ← level 파라미터 지원 버전
    └── main.py

모델 구성 (v10 구조 반영):
    M1. LightGBM  — v3 피처  (SEEDS_5 × 5-Fold)
    M2. LightGBM  — v3 + TE  (SEEDS_5 × 5-Fold)   ← 시술 시기 코드 Target Encoding
    M3. XGBoost   — v3 피처  (SEEDS_3 × 5-Fold)
    M4. XGBoost   — full 피처 (SEEDS_3 × 5-Fold)
    M5. LightGBM  — split 모델 (이식/비이식 분리, v3, SEEDS_5)
    M6. CatBoost  — full 피처 (SEEDS_3 × 5-Fold)
    ──────────────────────────────────────────────
    ENS. OOF 기반 Nelder-Mead 가중치 최적화
"""

import sys
import gc
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegressionCV
from catboost import CatBoostRegressor      # 추가
warnings.filterwarnings('ignore')

# =============================================================================
# 시각적 진행률 트래커 (외부 라이브러리 불필요)
# =============================================================================
class ProgressTracker:
    """
    터미널에 실시간 진행 상황 + 예상 잔여 시간 출력.
    
    사용법:
        tracker = ProgressTracker(total_steps=25, model_name='M1 LGB_v3')
        tracker.step(auc=0.7401)   # 매 fold 종료 시 호출
        tracker.done()             # 모델 완료 시 호출
    """
    BAR_WIDTH = 30

    def __init__(self, total_steps: int, model_name: str):
        self.total   = total_steps
        self.name    = model_name
        self.current = 0
        self.aucs    = []
        self.t_start = time.time()
        self.fold_times: list[float] = []
        self._print_header()

    def _bar(self, done: int) -> str:
        filled = int(self.BAR_WIDTH * done / max(self.total, 1))
        return '█' * filled + '░' * (self.BAR_WIDTH - filled)

    def _eta(self) -> str:
        if not self.fold_times:
            return '--:--'
        avg   = sum(self.fold_times) / len(self.fold_times)
        left  = avg * (self.total - self.current)
        m, s  = divmod(int(left), 60)
        h, m  = divmod(m, 60)
        if h:
            return f'{h}h {m:02d}m {s:02d}s'
        return f'{m:02d}m {s:02d}s'

    def _elapsed(self) -> str:
        sec  = int(time.time() - self.t_start)
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        if h:
            return f'{h}h {m:02d}m {s:02d}s'
        return f'{m:02d}m {s:02d}s'

    def _print_header(self):
        print(f'\n┌{"─"*60}┐')
        print(f'│  🚀  {self.name:<52}│')
        print(f'│  총 {self.total}회 (seed × fold){" "*(60-14-len(str(self.total)))}│')
        print(f'└{"─"*60}┘')

    def step(self, elapsed_fold: float, auc: float = None,
             seed: int = None, fold: int = None):
        self.current += 1
        self.fold_times.append(elapsed_fold)
        if auc is not None:
            self.aucs.append(auc)

        bar      = self._bar(self.current)
        pct      = 100 * self.current / self.total
        best_auc = f'{max(self.aucs):.5f}' if self.aucs else '------'
        eta_str  = self._eta()
        ela_str  = self._elapsed()

        info = ''
        if seed is not None and fold is not None:
            info = f'  seed={seed} fold{fold}'

        # \r 로 같은 줄 덮어쓰기 → 깔끔한 실시간 갱신
        line = (f'\r  [{bar}] {self.current:>3}/{self.total}'
                f'  {pct:5.1f}%'
                f'  AUC={best_auc}'
                f'  경과={ela_str}'
                f'  남은={eta_str}'
                f'{info:<20}')
        print(line, end='', flush=True)

    def seed_done(self, seed: int, seed_auc: float):
        """시드 완료 시 줄 바꿈 + 시드 결과 출력"""
        print(f'\n     └─ seed {seed} OOF = {seed_auc:.5f}', flush=True)

    def done(self, final_auc: float):
        """모델 완료"""
        ela = self._elapsed()
        print(f'\n  {"─"*58}')
        print(f'  ✅  {self.name} 완료 │ OOF AUC = {final_auc:.5f} │ 소요 {ela}')
        print(f'  {"─"*58}\n', flush=True)


class PipelineProgress:
    """전체 파이프라인 단계 표시기"""
    STAGES = [
        ('전처리 + FE',       '⚙️ '),
        ('M1 LGB_v3',         '🌿'),
        ('M2 LGB_v3+TE',      '🌿'),
        ('M3 XGB_v3',         '🌳'),
        ('M4 XGB_full',       '🌳'),
        ('M5 분리모델',        '✂️ '),
        ('M6 CatBoost',       '🐱'),
        ('M7 잔차보정',        '🔧'),
        ('앙상블 최적화',      '⚖️ '),
    ]

    def __init__(self):
        self.current  = 0
        self.t_global = time.time()
        self._banner()

    def _banner(self):
        total = len(self.STAGES)
        print('\n' + '=' * 62)
        print('  IVF 난임 예측  |  전체 파이프라인 진행 현황')
        print('=' * 62)
        for i, (name, icon) in enumerate(self.STAGES):
            mark = '⬜'
            print(f'  {mark}  {icon}  {i+1}. {name}')
        print('=' * 62)

    def advance(self, msg: str = ''):
        """다음 단계로 이동"""
        self.current += 1
        n     = len(self.STAGES)
        done  = self.current
        left  = n - done
        ela   = int(time.time() - self.t_global)
        m, s  = divmod(ela, 60)
        h, m2 = divmod(m, 60)
        ela_s = f'{h}h {m2:02d}m {s:02d}s' if h else f'{m2:02d}m {s:02d}s'

        bar_w  = 40
        filled = int(bar_w * done / n)
        bar    = '█' * filled + '░' * (bar_w - filled)

        print(f'\n  [{bar}] {done}/{n}  경과={ela_s}')
        name, icon = self.STAGES[self.current - 1]
        print(f'  {icon}  {name} {"완료" if not msg else msg}', flush=True)

    def finish(self, best_auc: float):
        ela   = int(time.time() - self.t_global)
        m, s  = divmod(ela, 60)
        h, m2 = divmod(m, 60)
        ela_s = f'{h}h {m2:02d}m {s:02d}s' if h else f'{m2:02d}m {s:02d}s'
        bar   = '█' * 40
        print(f'\n  [{bar}] 완료!')
        print(f'\n{"="*62}')
        print(f'  🏆  최종 앙상블 OOF AUC : {best_auc:.5f}')
        print(f'  ⏱️   총 소요 시간         : {ela_s}')
        print(f'{"="*62}\n', flush=True)



# ── 경로 ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
OUT_DIR  = BASE_DIR / 'output'
sys.path.insert(0, str(BASE_DIR))

from pipeline.schema import TARGET, DROP_COLUMNS, CENSORED_COLUMNS
from pipeline.preprocess import IVFPreprocessor
from pipeline.feature_engineering import IVFFeatureEngineer

# ── 설정 ──────────────────────────────────────────────────────────────────
SEED    = 42
N_FOLDS = 5
SEEDS_5 = [42, 2024, 777, 31415, 99999]   # LGB: 5 seeds × 5 fold = 25 모델
SEEDS_3 = [42, 2024, 777]                  # XGB/CAT: 3 seeds × 5 fold = 15 모델

TE_COL       = '시술 시기 코드'
TE_SMOOTHING = 20.0

# CatBoost에서 범주형으로 유지할 컬럼들
KEEP_CAT_COLS = [
    '시술 시기 코드', '시술 당시 나이', '시술 유형',
    '배란 유도 유형', '난자 출처', '정자 출처',
    '난자 기증자 나이', '정자 기증자 나이',
]
KEEP_CAT_FULL = KEEP_CAT_COLS + ['시술적합성', '배아수_그룹', 'ICSI_난자_구간']


# =============================================================================
# Step 0. 데이터 로드
# =============================================================================
def load_data():
    for p in [DATA_DIR / 'train.csv', DATA_DIR / 'test.csv',
              DATA_DIR / 'sample_submission.csv']:
        assert p.exists(), f"❌ 파일 없음: {p}"

    train = pd.read_csv(DATA_DIR / 'train.csv')
    test  = pd.read_csv(DATA_DIR / 'test.csv')
    sub   = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    print(f"✅ 데이터 로드  |  train {train.shape}  test {test.shape}")
    print(f"   성공률: {train[TARGET].mean():.4f}")
    return train, test, sub


# =============================================================================
# Step 1. 전처리 — IVFPreprocessor (fit: train only, Leakage 방지)
# =============================================================================
def run_preprocessing(train_raw, test_raw):
    preprocessor = IVFPreprocessor()
    X_pre      = preprocessor.fit_transform(train_raw.drop(columns=[TARGET]))
    X_test_pre = preprocessor.transform(test_raw)
    y          = train_raw[TARGET].astype(np.int8)
    print(f"✅ 전처리 완료  |  train {X_pre.shape}  test {X_test_pre.shape}")
    return X_pre, X_test_pre, y, preprocessor


# =============================================================================
# Step 2. 피처 엔지니어링 — IVFFeatureEngineer
# =============================================================================
def run_fe(X_pre, X_test_pre, level='full'):
    """IVFFeatureEngineer.transform(level=) 호출"""
    eng   = IVFFeatureEngineer()
    X_fe       = eng.transform(X_pre.copy(),      level=level)
    X_test_fe  = eng.transform(X_test_pre.copy(), level=level)
    print(f"✅ FE 완료 [{level}]  |  train {X_fe.shape}  test {X_test_fe.shape}")
    return X_fe, X_test_fe


# =============================================================================
# Step 3. Target Encoding — 시술 시기 코드 (OOF, Leakage-safe)
# =============================================================================
def _smoothed_mean(group_df, col, y_vals, smoothing):
    agg = pd.DataFrame({'k': group_df[col].values, 'y': y_vals})
    agg = agg.groupby('k')['y'].agg(['mean', 'count'])
    gm  = float(y_vals.mean())
    sm  = (agg['mean'] * agg['count'] + gm * smoothing) / (agg['count'] + smoothing)
    return sm.to_dict(), gm


def run_te(X, X_test, y, col=TE_COL, extra_keys=False):
    """
    ★ v10 반영: Smoothed TE 3종 콤보 키
      - period_te        : 시술 시기 코드 (smoothing=20)
      - TE__proc_specific: has_* 6개 토큰 조합 (smoothing=30)
      - TE__proc_age     : 시술유형 × 나이 (smoothing=25)
      - TE__proc_timing  : 시술유형 × 시기코드 (smoothing=25)

    누수 방지: train fold만으로 mapping → val/test 적용
    """
    if col not in X.columns:
        print(f"⚠️  {col} 없음 — TE 스킵")
        return X, X_test

    X      = X.copy()
    X_test = X_test.copy()

    gm  = float(y.mean())
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # ── 기본 TE: 시술 시기 코드 ────────────────────────────────────────
    oof_te = np.zeros(len(X), dtype=np.float32)
    for _, (tr_i, val_i) in enumerate(skf.split(X, y)):
        mapping, gm_ = _smoothed_mean(X.iloc[tr_i], col, y.iloc[tr_i], 20.0)
        oof_te[val_i] = X.iloc[val_i][col].map(mapping).fillna(gm_).values

    full_map, _ = _smoothed_mean(X, col, y, 20.0)
    X['period_te']      = oof_te
    X_test['period_te'] = X_test[col].map(full_map).fillna(gm).values.astype(np.float32)
    X      = X.drop(columns=[col])
    X_test = X_test.drop(columns=[col])
    print(f"   period_te 추가, {col} 제거")

    if not extra_keys:
        return X, X_test

    # ── ★ v10 콤보 TE 3종 ──────────────────────────────────────────────
    # 1. proc_specific: has_* 6개 토큰을 문자열로 이어 붙인 조합 키
    #    (특정 시술 유형의 정확한 조합별 성공률 인코딩)
    proc_tok_cols = [f'has_{t}' for t in ['ICSI','IVF','BLASTOCYST','AH','FER','Unknown']]
    avail_toks = [c for c in proc_tok_cols if c in X.columns]
    if avail_toks:
        key_train = X[avail_toks].astype(str).apply(''.join, axis=1)
        key_test  = X_test[avail_toks].astype(str).apply(''.join, axis=1)
        oof_ps = np.zeros(len(X), dtype=np.float32)
        full_map_ps, _ = _smoothed_mean(
            pd.DataFrame({'k': key_train.values}), 'k', y, 30.0
        )
        for _, (tr_i, val_i) in enumerate(skf.split(X, y)):
            m_ps, gm_ps = _smoothed_mean(
                pd.DataFrame({'k': key_train.iloc[tr_i].values}), 'k',
                y.iloc[tr_i], 30.0
            )
            oof_ps[val_i] = key_train.iloc[val_i].map(m_ps).fillna(gm_ps).values
        X['TE__proc_specific']      = oof_ps
        X_test['TE__proc_specific'] = key_test.map(full_map_ps).fillna(gm).values.astype(np.float32)

    # 2. proc_age: 시술유형 × 나이 조합 (smoothing=25)
    if '시술 유형' in X.columns and '시술 당시 나이' in X.columns:
        key_train = X['시술 유형'].astype(str) + '_' + X['시술 당시 나이'].astype(str)
        key_test  = X_test['시술 유형'].astype(str) + '_' + X_test['시술 당시 나이'].astype(str)
        oof_pa = np.zeros(len(X), dtype=np.float32)
        full_map_pa, _ = _smoothed_mean(
            pd.DataFrame({'k': key_train.values}), 'k', y, 25.0
        )
        for _, (tr_i, val_i) in enumerate(skf.split(X, y)):
            m_pa, gm_pa = _smoothed_mean(
                pd.DataFrame({'k': key_train.iloc[tr_i].values}), 'k',
                y.iloc[tr_i], 25.0
            )
            oof_pa[val_i] = key_train.iloc[val_i].map(m_pa).fillna(gm_pa).values
        X['TE__proc_age']      = oof_pa
        X_test['TE__proc_age'] = key_test.map(full_map_pa).fillna(gm).values.astype(np.float32)

    # 3. proc_timing: 시술유형 × 시기코드 조합 (smoothing=25)
    #    (시기코드는 위에서 period_te로 변환됐으므로 원본 backup 필요 없이
    #     시술유형과 나이 조합의 변형으로 대체 — 시술유형 × 시술당시나이 ordinal)
    if '시술 유형' in X.columns and 'age_ord' in X.columns:
        key_train = X['시술 유형'].astype(str) + '_' + X['age_ord'].astype(str)
        key_test  = X_test['시술 유형'].astype(str) + '_' + X_test['age_ord'].astype(str)
        oof_pt = np.zeros(len(X), dtype=np.float32)
        full_map_pt, _ = _smoothed_mean(
            pd.DataFrame({'k': key_train.values}), 'k', y, 25.0
        )
        for _, (tr_i, val_i) in enumerate(skf.split(X, y)):
            m_pt, gm_pt = _smoothed_mean(
                pd.DataFrame({'k': key_train.iloc[tr_i].values}), 'k',
                y.iloc[tr_i], 25.0
            )
            oof_pt[val_i] = key_train.iloc[val_i].map(m_pt).fillna(gm_pt).values
        X['TE__proc_timing']      = oof_pt
        X_test['TE__proc_timing'] = key_test.map(full_map_pt).fillna(gm).values.astype(np.float32)

    added = [c for c in ['TE__proc_specific','TE__proc_age','TE__proc_timing'] if c in X.columns]
    print(f"   콤보 TE 추가: {added}")
    return X, X_test


# =============================================================================
# Step 4. 피처 확정 — Label Encoding (train fit, Leakage 방지)
# =============================================================================
def finalize(X, X_test, for_catboost=False):
    """
    str/object 컬럼 처리:
      for_catboost=False: Label Encoding (LGB/XGB용)
      for_catboost=True : string 유지, 수치 문자열만 변환 (CatBoost용)
    """
    import re

    shared = [c for c in X.columns if c in X_test.columns]
    X      = X[shared].copy()
    X_test = X_test[shared].copy()

    def to_num(val):
        if pd.isna(val) or str(val) in ('nan', 'None', '알 수 없음'):
            return np.nan
        m = re.search(r'\d+', str(val))
        return float(m.group()) if m else np.nan

    str_cols = [c for c in X.columns
                if X[c].dtype == 'object' or str(X[c].dtype) in ('string', 'str')]

    cat_encoded, num_converted = [], []
    for col in str_cols:
        sample = X[col].dropna().astype(str).head(30)
        is_count = sample.str.contains(r'\d+회', regex=True).any()

        if is_count:
            X[col]      = pd.to_numeric(X[col].apply(to_num), errors='coerce')
            X_test[col] = pd.to_numeric(X_test[col].apply(to_num), errors='coerce')
            num_converted.append(col)
        elif for_catboost:
            for df in [X, X_test]:
                df[col] = df[col].fillna('NaN').astype(str)
        else:
            le = LabelEncoder()
            all_vals = sorted(set(
                X[col].fillna('missing').astype(str).tolist() +
                X_test[col].fillna('missing').astype(str).tolist()
            ))
            le.fit(all_vals)
            X[col]      = le.transform(X[col].fillna('missing').astype(str))
            X_test[col] = le.transform(X_test[col].fillna('missing').astype(str))
            cat_encoded.append(col)

    for df in [X, X_test]:
        for c in df.select_dtypes(include=[np.number]).columns:
            if df[c].isna().sum():
                df[c] = df[c].fillna(df[c].median() if not df[c].isna().all() else 0)

    print(f"✅ finalize 완료  |  {len(shared)}개 컬럼  "
          f"(LE {len(cat_encoded)}, num변환 {len(num_converted)})")
    return X, X_test, cat_encoded


# =============================================================================
# 피처셋 사전 준비 (전처리 1회 → 3가지 변형)
# =============================================================================
def prepare_feature_sets(X_pre, X_test_pre, y):
    """
    M1/M3/M5: v3 피처셋
    M2:        v3 + Target Encoding
    M4/M6:     full 피처셋

    반환: dict { 'v3': (...), 'v3_te': (...), 'full': (...), 'full_cat': (...) }
    """
    print("\n[피처셋 준비]")

    # v3 피처셋
    Xv3, Xv3t = run_fe(X_pre, X_test_pre, level='v3')
    Xv3_enc, Xv3t_enc, cat_v3 = finalize(Xv3, Xv3t)

    # v3 + TE (M2용) — TE 먼저, finalize 나중
    Xv3_te, Xv3t_te = run_fe(X_pre, X_test_pre, level='v3')
    Xv3_te, Xv3t_te = run_te(Xv3_te, Xv3t_te, y, extra_keys=True)
    Xv3te_enc, Xv3te_t_enc, cat_v3te = finalize(Xv3_te, Xv3t_te)

    # full 피처셋 (LGB/XGB용, Label Encoding)
    Xf, Xft = run_fe(X_pre, X_test_pre, level='full')
    Xf_enc, Xft_enc, cat_full = finalize(Xf, Xft)

    # full 피처셋 (CatBoost용, string 유지)
    Xfc, Xftc = run_fe(X_pre, X_test_pre, level='full')
    Xfc_enc, Xftc_enc, _ = finalize(Xfc, Xftc, for_catboost=True)
    # KEEP_CAT_FULL 목록 대신 finalize 후 실제 남아있는
    # 모든 string 컬럼을 자동으로 cat_features에 포함
    cat_idx_full = [
        i for i, c in enumerate(Xfc_enc.columns)
        if Xfc_enc[c].dtype == 'object'
        or str(Xfc_enc[c].dtype) in ('string', 'str')
    ]
    
    return {
        'v3':       (Xv3_enc,   Xv3t_enc,   cat_v3),
        'v3_te':    (Xv3te_enc, Xv3te_t_enc, cat_v3te),
        'full':     (Xf_enc,    Xft_enc,     cat_full),
        'full_cat': (Xfc_enc,   Xftc_enc,    cat_idx_full),
    }


# =============================================================================
# 공통 K-Fold 학습 루프 (Multi-seed, Early Stopping 지원)
# =============================================================================
def kfold_train(model_fn, X, X_test, y, model_name, seeds):
    """
    model_fn(X_tr, y_tr, X_val, y_val, X_test, seed) → (oof_fold, test_fold)
    """
    oof_pred  = np.zeros(len(X),      dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)

    total_steps = len(seeds) * N_FOLDS
    tracker = ProgressTracker(total_steps=total_steps, model_name=model_name)

    for seed in seeds:
        skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof_s = np.zeros(len(X),      dtype=np.float64)
        pre_s = np.zeros(len(X_test), dtype=np.float64)

        for fold, (tri, vai) in enumerate(skf.split(X, y)):
            t0 = time.time()
            fold_oof, fold_test = model_fn(
                X.iloc[tri], y.iloc[tri],
                X.iloc[vai], y.iloc[vai],
                X_test, seed
            )
            oof_s[vai] = fold_oof
            pre_s     += fold_test / N_FOLDS
            auc        = roc_auc_score(y.iloc[vai], fold_oof)
            tracker.step(elapsed_fold=time.time()-t0, auc=auc,
                         seed=seed, fold=fold+1)

        seed_auc = roc_auc_score(y, oof_s)
        tracker.seed_done(seed=seed, seed_auc=seed_auc)
        oof_pred  += oof_s
        test_pred += pre_s
        del oof_s, pre_s; gc.collect()

    oof_pred  /= len(seeds)
    test_pred /= len(seeds)
    oof_auc    = roc_auc_score(y, oof_pred)
    tracker.done(final_auc=oof_auc)
    return oof_pred, test_pred, oof_auc


# =============================================================================
# 모델 파라미터 & fold 함수
# =============================================================================

LGB_BASE = dict(
    objective='binary', metric='auc',
    learning_rate=0.025,
    num_leaves=95, max_depth=-1,
    feature_fraction=0.8,
    bagging_fraction=0.85, bagging_freq=1,
    min_data_in_leaf=150,
    lambda_l1=0.05, lambda_l2=1.0,
    verbose=-1, n_jobs=-1,
)

XGB_BASE = dict(
    objective='binary:logistic', eval_metric='auc',
    learning_rate=0.03,
    max_depth=7, min_child_weight=10,
    subsample=0.85, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    tree_method='hist', verbosity=0, n_jobs=-1,
)

CAT_BASE = dict(
    iterations=600,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    loss_function='Logloss',
    eval_metric='AUC',
    early_stopping_rounds=60,   # ★ v10 동일: 100→60
    verbose=0,
    task_type='CPU',
    bootstrap_type='Bernoulli',
    subsample=0.7,
    rsm=0.7,
    max_ctr_complexity=0,       # ★ v10 동일: 2→0 (과적합 방지)
    border_count=64,
    thread_count=-1,
)


def lgb_fold_fn(cat_cols):
    """LightGBM fold 함수 생성기 (early stopping 포함)"""
    import lightgbm as lgb

    def fn(X_tr, y_tr, X_val, y_val, X_te, seed):
        params = {
            **LGB_BASE,
            'random_state': seed, 'seed': seed,
            'bagging_seed': seed, 'feature_fraction_seed': seed,
        }
        dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols or 'auto',
                          free_raw_data=True)
        dva = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols or 'auto',
                          reference=dtr, free_raw_data=True)
        m = lgb.train(
            params, dtr,
            num_boost_round=5000,
            valid_sets=[dva], valid_names=['v'],
            callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)]
        )
        oof  = m.predict(X_val, num_iteration=m.best_iteration)
        test = m.predict(X_te,  num_iteration=m.best_iteration)
        del m, dtr, dva; gc.collect()
        return oof, test

    return fn


def xgb_fold_fn(enable_cat=False):
    import xgboost as xgb

    def _to_cat(df):
        if not enable_cat:
            return df
        df = df.copy()
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].astype('category')
        return df

    def fn(X_tr, y_tr, X_val, y_val, X_te, seed):
        params = {**XGB_BASE, 'seed': seed}
        dtr = xgb.DMatrix(_to_cat(X_tr),  label=y_tr,  enable_categorical=enable_cat)
        dva = xgb.DMatrix(_to_cat(X_val), label=y_val, enable_categorical=enable_cat)
        dte = xgb.DMatrix(_to_cat(X_te),               enable_categorical=enable_cat)
        m = xgb.train(
            params, dtr,
            num_boost_round=3000,
            evals=[(dva, 'v')], early_stopping_rounds=100, verbose_eval=False
        )
        oof  = m.predict(dva)
        test = m.predict(dte)
        del m, dtr, dva, dte; gc.collect()
        return oof, test

    return fn


def cat_fold_fn(cat_idx):
    """CatBoost fold 함수 생성기"""
    from catboost import CatBoostClassifier, Pool

    def fn(X_tr, y_tr, X_val, y_val, X_te, seed):
        params = {**CAT_BASE, 'random_seed': seed}
        tr_pool  = Pool(X_tr,  y_tr,  cat_features=cat_idx)
        val_pool = Pool(X_val, y_val, cat_features=cat_idx)
        m = CatBoostClassifier(**params)
        m.fit(tr_pool, eval_set=val_pool, use_best_model=True, verbose=0)
        oof  = m.predict_proba(X_val)[:, 1]
        test = m.predict_proba(X_te)[:, 1]
        del m, tr_pool, val_pool; gc.collect()
        return oof, test

    return fn

# [함수 추가] M7 잔차 학습 로직
def train_residual_model(X, X_test, y, base_oof, cat_idx):
    # ★ 수정: y.values 사용하여 pandas 인덱스 미스매치 방지
    residual = pd.Series(y.values - base_oof, index=y.index)
    oof_res = np.zeros(len(X))
    pred_res = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    print(f"\n[M7 잔차 보정] 오답 분석 중...")
    for tri, vai in skf.split(X, y):
        m = CatBoostRegressor(iterations=800, learning_rate=0.03, depth=6, verbose=0)
        m.fit(X.iloc[tri], residual.iloc[tri], eval_set=(X.iloc[vai], residual.iloc[vai]),
              cat_features=cat_idx, early_stopping_rounds=50)
        oof_res[vai] = m.predict(X.iloc[vai])
        pred_res += m.predict(X_test) / N_FOLDS
    return oof_res, pred_res

# [함수 추가] Ridge Stacking 로직
# ✅ 수정 (LogisticRegressionCV로 교체 — 분류 문제에 적합)

def run_ridge_stacking(oofs, preds, y):
    keys = list(oofs.keys())
    X_meta      = np.stack([oofs[k]  for k in keys], axis=1)
    X_test_meta = np.stack([preds[k] for k in keys], axis=1)

    # RidgeCV → LogisticRegressionCV
    # 이진 분류이므로 확률값(predict_proba)을 직접 출력 → AUC에 최적
    meta_model = LogisticRegressionCV(
        Cs=[0.01, 0.1, 1.0, 10.0],
        cv=5,
        scoring='roc_auc',
        max_iter=1000,
        random_state=42,
    )
    meta_model.fit(X_meta, y)

    final_oof  = meta_model.predict_proba(X_meta)[:, 1]
    final_pred = meta_model.predict_proba(X_test_meta)[:, 1]
    stack_auc  = roc_auc_score(y, final_oof)
    return final_pred, stack_auc

# =============================================================================
# 분리 모델 (M5) — 이식 / 비이식 별도 학습
# =============================================================================
def run_split_model(X, X_test, y, cat_cols, seeds=SEEDS_5):
    """
    이식된 배아 수 > 0 여부로 분리 후 각각 LGB 학습.
    OOF 예측을 원래 인덱스 위치에 채워 넣는다.

    누수 방지: 분리 기준은 이식 여부(이진)이며 타겟 통계 미사용.
    """
    import lightgbm as lgb

    transfer_col = '이식된 배아 수'
    if transfer_col not in X.columns:
        print("⚠️  이식된 배아 수 컬럼 없음 — 분리 모델 스킵")
        return None, None, None

    tr_mask   = (pd.to_numeric(X[transfer_col], errors='coerce').fillna(0) > 0).values
    te_mask   = (pd.to_numeric(X_test[transfer_col], errors='coerce').fillna(0) > 0).values

    print(f"\n[M5 분리 모델]  이식 {tr_mask.sum():,}건 / 비이식 {(~tr_mask).sum():,}건")

    oof_pred  = np.zeros(len(X),      dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)

    for group_name, g_tr, g_te, mil in [
        ('NoTransfer', ~tr_mask, ~te_mask, 100),
        ('Transfer',    tr_mask,  te_mask, 200),
    ]:
        sub_X    = X[g_tr].reset_index(drop=True)
        sub_y    = y[g_tr].reset_index(drop=True)
        sub_Xte  = X_test[g_te].reset_index(drop=True)

        oof_g    = np.zeros(len(sub_X),   dtype=np.float64)
        pred_g   = np.zeros(len(sub_Xte), dtype=np.float64)

        lgb_p    = {**LGB_BASE, 'min_data_in_leaf': mil}
        tracker  = ProgressTracker(
            total_steps=len(seeds) * N_FOLDS,
            model_name=f'M5 분리모델 [{group_name}]'
        )

        for seed in seeds:
            skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            oof_s = np.zeros(len(sub_X),   dtype=np.float64)
            pre_s = np.zeros(len(sub_Xte), dtype=np.float64)

            for fold, (tri, vai) in enumerate(skf.split(sub_X, sub_y)):
                t0 = time.time()
                params = {
                    **lgb_p,
                    'random_state': seed, 'seed': seed,
                    'bagging_seed': seed, 'feature_fraction_seed': seed,
                }
                dtr = lgb.Dataset(sub_X.iloc[tri], label=sub_y.iloc[tri],
                                  categorical_feature=cat_cols or 'auto', free_raw_data=True)
                dva = lgb.Dataset(sub_X.iloc[vai], label=sub_y.iloc[vai],
                                  categorical_feature=cat_cols or 'auto',
                                  reference=dtr, free_raw_data=True)
                m = lgb.train(
                    params, dtr, num_boost_round=5000,
                    valid_sets=[dva], valid_names=['v'],
                    callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)]
                )
                fold_auc = roc_auc_score(sub_y.iloc[vai],
                                         m.predict(sub_X.iloc[vai], num_iteration=m.best_iteration))
                oof_s[vai] = m.predict(sub_X.iloc[vai], num_iteration=m.best_iteration)
                pre_s     += m.predict(sub_Xte, num_iteration=m.best_iteration) / N_FOLDS
                tracker.step(elapsed_fold=time.time()-t0, auc=fold_auc,
                             seed=seed, fold=fold+1)
                del m, dtr, dva; gc.collect()

            seed_auc_g = roc_auc_score(sub_y, oof_s)
            tracker.seed_done(seed=seed, seed_auc=seed_auc_g)

            oof_g  += oof_s
            pred_g += pre_s
            del oof_s, pre_s; gc.collect()

        oof_g  /= len(seeds)
        pred_g /= len(seeds)
        oof_pred[g_tr]  = oof_g
        test_pred[g_te] = pred_g
        auc_g = roc_auc_score(sub_y, oof_g)
        print(f"   [{group_name}] OOF AUC={auc_g:.5f}")

    oof_auc = roc_auc_score(y, oof_pred)
    print(f"   ─────────────────────────────")
    print(f"   M5 분리 모델 OOF AUC: {oof_auc:.5f}")
    return oof_pred, test_pred, oof_auc


# =============================================================================
# 앙상블 — OOF 기반 Nelder-Mead 가중치 탐색
# =============================================================================
def run_ensemble(oofs: dict, preds: dict, y):
    """
    누수 방지:
      - 가중치 탐색은 OOF 예측에서만 수행
      - test 예측은 최적 가중치 확정 후 선형 결합
      - 5개 시작점에서 탐색 → 전역 최적 확보
    """
    keys     = list(oofs.keys())
    oof_arr  = np.stack([oofs[k]  for k in keys], axis=1).astype(np.float64)
    pred_arr = np.stack([preds[k] for k in keys], axis=1).astype(np.float64)

    def neg_auc(w):
        w = np.clip(w, 0, None)
        s = w.sum()
        return 0.0 if s < 1e-9 else -roc_auc_score(y, oof_arr @ (w / s))

    starts = [
        [0.0, 0.31, 0.165, 0.387, 0.138, 0.0],
        [0.0, 0.25, 0.15,  0.30,  0.10,  0.20],
        [0.0, 0.20, 0.15,  0.25,  0.10,  0.30],
        [0.1, 0.20, 0.15,  0.25,  0.10,  0.20],
        [1 / len(keys)] * len(keys),
    ]

    print(f"\n[앙상블]  {keys}")
    best_w, best_auc = None, -np.inf
    for st in starts:
        st = (list(st) + [1/len(keys)] * len(keys))[:len(keys)]
        res = minimize(neg_auc, np.array(st), method='Nelder-Mead',
                       options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-7})
        w   = np.clip(res.x, 0, None); w /= w.sum()
        auc = roc_auc_score(y, oof_arr @ w)
        if auc > best_auc:
            best_auc, best_w = auc, w

    print("\n   개별 OOF AUC:")
    for k in keys:
        print(f"     {k:15s}  {roc_auc_score(y, oofs[k]):.5f}")
    print("\n   최적 가중치:")
    for k, v in zip(keys, best_w):
        print(f"     {k:15s}  {v:.4f}")
    print(f"\n   앙상블 OOF AUC: {best_auc:.5f}  ★")

    return oof_arr @ best_w, pred_arr @ best_w, best_w


# =============================================================================
# 제출 파일 저장
# =============================================================================
def save_submission(test_pred, sub_template, oof_auc, suffix='ensemble'):
    OUT_DIR.mkdir(exist_ok=True)
    sub      = sub_template.copy()
    prob_col = [c for c in sub.columns if c != 'ID'][0]
    sub[prob_col] = np.clip(test_pred, 0.0, 1.0)
    ts    = datetime.now().strftime('%m%d_%H%M')
    fname = f'submission_{suffix}_AUC{oof_auc:.5f}_{ts}.csv'
    sub.to_csv(OUT_DIR / fname, index=False)
    print(f"   💾 {OUT_DIR / fname}")
    print(f"      mean={sub[prob_col].mean():.4f}  std={sub[prob_col].std():.4f}")


# =============================================================================
# 메인 실행
# =============================================================================
def main():
    print("=" * 62)
    print("IVF 난임 예측  (6-Model Ensemble, pipeline 기반)")
    print("=" * 62)
    OUT_DIR.mkdir(exist_ok=True)

    pipeline = PipelineProgress()   # 전체 파이프라인 진행 표시

    # 0. 로드
    train_raw, test_raw, sub_template = load_data()

    # 1. 전처리 (1회, Leakage-safe)
    X_pre, X_test_pre, y, _ = run_preprocessing(train_raw, test_raw)

    # 2. 피처셋 3종 준비
    print("\n" + "─" * 62)
    feat = prepare_feature_sets(X_pre, X_test_pre, y)
    pipeline.advance('완료')   # ⚙️  전처리 + FE

    X_v3,  X_v3t,  cat_v3   = feat['v3']
    X_te,  X_tet,  cat_te   = feat['v3_te']
    X_fl,  X_flt,  cat_fl   = feat['full']
    X_fc,  X_fct,  cat_cidx = feat['full_cat']

    oofs, preds = {}, {}

    # ── M1. LightGBM v3 ──────────────────────────────────────────────────
    print("\n" + "─" * 62)
    oof, pred, auc = kfold_train(
        lgb_fold_fn(cat_v3), X_v3, X_v3t, y, 'M1 LGB_v3', SEEDS_5
    )
    np.save(OUT_DIR / 'oof_m1_lgb_v3.npy',  oof)
    np.save(OUT_DIR / 'pred_m1_lgb_v3.npy', pred)
    oofs['LGB_v3'] = oof; preds['LGB_v3'] = pred
    pipeline.advance(f'OOF AUC={auc:.5f}')   # 🌿 M1

    # ── M2. LightGBM v3 + TE ─────────────────────────────────────────────
    print("\n" + "─" * 62)
    oof, pred, auc = kfold_train(
        lgb_fold_fn(cat_te), X_te, X_tet, y, 'M2 LGB_v3TE', SEEDS_5
    )
    np.save(OUT_DIR / 'oof_m2_lgb_v3te.npy',  oof)
    np.save(OUT_DIR / 'pred_m2_lgb_v3te.npy', pred)
    oofs['LGB_v3TE'] = oof; preds['LGB_v3TE'] = pred
    pipeline.advance(f'OOF AUC={auc:.5f}')   # 🌿 M2

    # ── M3. XGBoost v3 ───────────────────────────────────────────────────
    print("\n" + "─" * 62)
    oof, pred, auc = kfold_train(
        xgb_fold_fn(), X_v3, X_v3t, y, 'M3 XGB_v3', SEEDS_3
    )
    np.save(OUT_DIR / 'oof_m3_xgb_v3.npy',  oof)
    np.save(OUT_DIR / 'pred_m3_xgb_v3.npy', pred)
    oofs['XGB_v3'] = oof; preds['XGB_v3'] = pred
    pipeline.advance(f'OOF AUC={auc:.5f}')   # 🌳 M3

    # ── M4. XGBoost full ─────────────────────────────────────────────────
    print("\n" + "─" * 62)
    oof, pred, auc = kfold_train(
        xgb_fold_fn(enable_cat=True), X_fc, X_fct, y, 'M4 XGB_full', SEEDS_3
    )
    np.save(OUT_DIR / 'oof_m4_xgb_full.npy',  oof)
    np.save(OUT_DIR / 'pred_m4_xgb_full.npy', pred)
    oofs['XGB_full'] = oof; preds['XGB_full'] = pred
    pipeline.advance(f'OOF AUC={auc:.5f}')   # 🌳 M4

    # ── M5. 분리 모델 (이식/비이식, v3) ─────────────────────────────────
    print("\n" + "─" * 62)
    oof, pred, auc = run_split_model(X_v3, X_v3t, y, cat_v3, seeds=SEEDS_5)
    if oof is not None:
        np.save(OUT_DIR / 'oof_m5_split.npy',  oof)
        np.save(OUT_DIR / 'pred_m5_split.npy', pred)
        oofs['split'] = oof; preds['split'] = pred
    pipeline.advance(f'OOF AUC={auc:.5f}' if oof is not None else '스킵')

    # ── M6. CatBoost full ────────────────────────────────────────────────
    print("\n" + "─" * 62)
    oof, pred, auc = kfold_train(
        cat_fold_fn(cat_cidx), X_fc, X_fct, y, 'M6 CatBoost', SEEDS_3
    )
    np.save(OUT_DIR / 'oof_m6_catboost.npy',  oof)
    np.save(OUT_DIR / 'pred_m6_catboost.npy', pred)
    oofs['CatBoost'] = oof; preds['CatBoost'] = pred
    pipeline.advance(f'OOF AUC={auc:.5f}')   # 🐱 M6

    # ── M7 잔차 보정 ────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("✨ [Phase 3] M7 잔차 학습 및 Ridge Stacking 가동")
    base_avg_oof  = np.mean([oofs[k]  for k in oofs.keys()], axis=0)
    base_avg_pred = np.mean([preds[k] for k in preds.keys()], axis=0)
    oof_res, pred_res = train_residual_model(X_fc, X_fct, y, base_avg_oof, cat_cidx)
    oofs['M7_Residual']  = np.clip(base_avg_oof  + oof_res,  0, 1)
    preds['M7_Residual'] = np.clip(base_avg_pred + pred_res, 0, 1)
    m7_auc = roc_auc_score(y, oofs['M7_Residual'])
    print(f"   - M7 보정 단독 AUC: {m7_auc:.5f}")
    pipeline.advance(f'보정 AUC={m7_auc:.5f}')   # 🔧 M7

    # ── Ridge Stacking + Nelder-Mead ─────────────────────────────────────
    final_pred_stack, stack_auc = run_ridge_stacking(oofs, preds, y)
    print(f"   - Ridge Stacking OOF AUC: {stack_auc:.5f} ★")
    save_submission(final_pred_stack, sub_template, stack_auc, 'RidgeStack')

    print("\n" + "=" * 62)
    print("⚖️  [Phase 4] 기존 Nelder-Mead 가중치 최적화 (M1~M7)")
    oof_ens, pred_ens, weights = run_ensemble(oofs, preds, y)
    ens_auc = roc_auc_score(y, oof_ens)
    np.save(OUT_DIR / 'oof_ensemble.npy',  oof_ens)
    np.save(OUT_DIR / 'pred_ensemble.npy', pred_ens)
    save_submission(pred_ens, sub_template, ens_auc, 'Ensemble_M7')
    pipeline.advance(f'앙상블 AUC={ens_auc:.5f}')  # ⚖️  앙상블

    # ── 중간 산출물 ──────────────────────────────────────────────────────
    X_v3.assign(**{TARGET: y.values}).to_csv(OUT_DIR / 'train_v3.csv',   index=False)
    X_fl.assign(**{TARGET: y.values}).to_csv(OUT_DIR / 'train_full.csv', index=False)
    X_v3t.to_csv(OUT_DIR / 'test_v3.csv',   index=False)
    X_flt.to_csv(OUT_DIR  / 'test_full.csv', index=False)

    best_auc = max(stack_auc, ens_auc)
    pipeline.finish(best_auc=best_auc)
    print(f"최종 결과: RidgeStack({stack_auc:.5f}) vs Ensemble_M7({ens_auc:.5f})")

    return oofs, preds, oof_ens, pred_ens


if __name__ == '__main__':
    main()