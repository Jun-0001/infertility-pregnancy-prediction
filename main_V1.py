"""
main_v11.py — IVF 난임 예측 단일 파일 (v10 구조 + 신규 피처)

v10 베이스 (0.74222) + 임상 신규 피처:
  - miscarriage_rate / has_miscarriage (유산률)
  - infertility_complexity (불임 원인 복합도)
  - is_elective_SET (선택적 단일 배아 이식)
  - high_ovarian_response (과자극 위험군 >20개)
  - is_blastocyst_by_day (경과일 기반 배반포 이식)
  - icsi_above_avg_success (ICSI 수정률 우수 여부)
  - is_DI (DI 시술 플래그)

모델 구성 (v10 동일):
  M1. LightGBM  v3       (SEEDS_5 × 5-Fold)
  M2. LightGBM  v3 + TE  (SEEDS_5 × 5-Fold)
  M3. XGBoost   v3feat   (SEEDS_3 × 5-Fold)
  M4. XGBoost   v11      (SEEDS_3 × 5-Fold)
  M5. LightGBM  split    (이식/비이식 분리, SEEDS_5)
  M6. CatBoost  v11      (SEEDS_3 × 5-Fold)
  ENS. OOF 기반 Nelder-Mead 가중치 최적화

폴더 구조:
  project/
  ├── data/
  │   ├── train.csv / test.csv / sample_submission.csv
  ├── output/
  └── main_v11.py
"""

import gc
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
OUT_DIR  = BASE_DIR / 'output'
OUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 상수 & 매핑
# ─────────────────────────────────────────────────────────────────────────────
TARGET   = '임신 성공 여부'
ID_COL   = 'ID'
N_FOLDS  = 5
SEEDS_5  = [42, 2024, 777, 31415, 99999]
SEEDS_3  = [42, 2024, 777]

age_map = {
    '만18-34세': 26.0, '만35-37세': 36.0, '만38-39세': 38.5,
    '만40-42세': 41.0, '만43-44세': 43.5, '만45-50세': 47.5,
    '알 수 없음': np.nan,
}
donor_age_map = {
    '만20세 이하': 19.0, '만21-25세': 23.0, '만26-30세': 28.0,
    '만31-35세': 33.0,   '만36-40세': 38.0, '만41-45세': 43.0,
    '알 수 없음': np.nan,
}
count_cols = [
    '총 시술 횟수', '클리닉 내 총 시술 횟수', 'IVF 시술 횟수', 'DI 시술 횟수',
    '총 임신 횟수', 'IVF 임신 횟수', 'DI 임신 횟수',
    '총 출산 횟수', 'IVF 출산 횟수', 'DI 출산 횟수',
]
_count_map = {
    '0회': 0.0, '1회': 1.0, '2회': 2.0, '3회': 3.0,
    '4회': 4.0, '5회': 5.0, '6회 이상': 6.0,
}
proc_tokens = ['ICSI', 'IVF', 'BLASTOCYST', 'AH', 'Unknown', 'FER']
reasons     = ['기증용', '난자 저장용', '배아 저장용', '현재 시술용']

KEEP_CAT_COLS = [
    '시술 시기 코드', '시술 당시 나이', '시술 유형',
    '배란 유도 유형', '난자 출처', '정자 출처',
    '난자 기증자 나이', '정자 기증자 나이',
]
KEEP_CAT_V11 = ['시술적합성', '배아수_그룹', 'ICSI_난자_구간']

# LGB 파라미터 (v10 동일)
LGB_PARAMS = dict(
    objective='binary', metric='auc',
    learning_rate=0.025,
    num_leaves=95, max_depth=-1,
    feature_fraction=0.8,
    bagging_fraction=0.85, bagging_freq=1,
    min_data_in_leaf=150,
    lambda_l1=0.05, lambda_l2=1.0,
    verbose=-1, n_jobs=-1,
)
# XGB 파라미터 (v10 동일)
XGB_PARAMS = dict(
    objective='binary:logistic', eval_metric='auc',
    learning_rate=0.03, max_depth=7,
    min_child_weight=10,
    subsample=0.85, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    tree_method='hist', device='cpu',
    n_jobs=-1, verbosity=0,
)
# CatBoost 파라미터 (v10 동일 — max_ctr_complexity=0)
CAT_PARAMS = dict(
    iterations=600,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    loss_function='Logloss',
    eval_metric='AUC',
    early_stopping_rounds=60,
    verbose=0,
    task_type='CPU',
    bootstrap_type='Bernoulli',
    subsample=0.7,
    rsm=0.7,
    max_ctr_complexity=0,   # v10 동일: 과적합 방지
    border_count=64,
    thread_count=-1,
)


# =============================================================================
# 시각적 진행 트래커
# =============================================================================
class ProgressTracker:
    BAR_W = 32

    def __init__(self, total: int, name: str):
        self.total      = total
        self.name       = name
        self.current    = 0
        self.aucs: list = []
        self.t_start    = time.time()
        self.fold_times: list = []
        print(f'\n┌{"─"*60}┐')
        print(f'│  🚀  {name:<52}│')
        print(f'│  총 {total}회 (seed × fold){" "*(60-14-len(str(total)))}│')
        print(f'└{"─"*60}┘')

    def _eta(self) -> str:
        if not self.fold_times:
            return '--:--'
        avg  = sum(self.fold_times) / len(self.fold_times)
        left = avg * (self.total - self.current)
        m, s = divmod(int(left), 60)
        h, m = divmod(m, 60)
        return f'{h}h {m:02d}m {s:02d}s' if h else f'{m:02d}m {s:02d}s'

    def _elapsed(self) -> str:
        sec  = int(time.time() - self.t_start)
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return f'{h}h {m:02d}m {s:02d}s' if h else f'{m:02d}m {s:02d}s'

    def step(self, t_fold: float, auc: float, seed: int, fold: int):
        self.current += 1
        self.fold_times.append(t_fold)
        self.aucs.append(auc)
        filled = int(self.BAR_W * self.current / self.total)
        bar    = '█' * filled + '░' * (self.BAR_W - filled)
        best   = f'{max(self.aucs):.5f}'
        line   = (f'\r  [{bar}] {self.current:>3}/{self.total}'
                  f'  {100*self.current/self.total:5.1f}%'
                  f'  best={best}'
                  f'  경과={self._elapsed()}'
                  f'  남은={self._eta()}'
                  f'  sd={seed} f{fold}   ')
        print(line, end='', flush=True)

    def seed_done(self, seed: int, auc: float):
        print(f'\n     └─ seed {seed}  OOF={auc:.5f}', flush=True)

    def done(self, final_auc: float):
        print(f'\n  {"─"*58}')
        print(f'  ✅  {self.name}  │  OOF AUC = {final_auc:.5f}  │  {self._elapsed()}')
        print(f'  {"─"*58}\n', flush=True)


class PipelineBar:
    STAGES = [
        ('데이터 로드 + FE',    '⚙️ '),
        ('M1 LGB v3',          '🌿'),
        ('M2 LGB v3+TE',       '🌿'),
        ('M3 XGB v3feat',      '🌳'),
        ('M4 XGB v11',         '🌳'),
        ('M5 분리모델',         '✂️ '),
        ('M6 CatBoost v11',    '🐱'),
        ('앙상블 최적화',       '⚖️ '),
    ]

    def __init__(self):
        self.idx     = 0
        self.t_start = time.time()
        n = len(self.STAGES)
        print('\n' + '=' * 62)
        print('  IVF 난임 예측 v11  |  파이프라인 진행 현황')
        print('=' * 62)
        for i, (name, icon) in enumerate(self.STAGES):
            print(f'  ⬜  {icon}  {i+1}. {name}')
        print('=' * 62)

    def advance(self, note: str = ''):
        self.idx += 1
        ela  = int(time.time() - self.t_start)
        m, s = divmod(ela, 60)
        h, m = divmod(m, 60)
        ela_s = f'{h}h {m:02d}m {s:02d}s' if h else f'{m:02d}m {s:02d}s'
        n      = len(self.STAGES)
        filled = int(40 * self.idx / n)
        bar    = '█' * filled + '░' * (40 - filled)
        name, icon = self.STAGES[self.idx - 1]
        print(f'\n  [{bar}] {self.idx}/{n}  경과={ela_s}')
        print(f'  {icon}  {name}  {note}', flush=True)

    def finish(self, best_auc: float):
        ela  = int(time.time() - self.t_start)
        m, s = divmod(ela, 60)
        h, m = divmod(m, 60)
        ela_s = f'{h}h {m:02d}m {s:02d}s' if h else f'{m:02d}m {s:02d}s'
        print(f'\n  [{"█"*40}] 완료!')
        print(f'\n{"="*62}')
        print(f'  🏆  최종 앙상블 OOF AUC : {best_auc:.5f}')
        print(f'  ⏱️   총 소요 시간         : {ela_s}')
        print(f'{"="*62}\n', flush=True)


# =============================================================================
# 피처 엔지니어링
# =============================================================================
def process_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """v10 동일: 모든 모델 공통 기반 피처 생성"""
    # [1] 횟수 컬럼 → 수치 변환 ('6회 이상' → 6.0)
    for c in count_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).replace('nan', np.nan).map(_count_map).astype('float32')

    # [2] 나이 → 수치
    df['age_num']           = df['시술 당시 나이'].map(age_map).astype('float32')
    df['egg_donor_age_num'] = df['난자 기증자 나이'].map(donor_age_map).astype('float32')
    df['sperm_donor_age_num'] = df['정자 기증자 나이'].map(donor_age_map).astype('float32')

    # [3] 특정 시술 유형 토큰 분해 (→ 원본 컬럼 drop)
    s = df['특정 시술 유형'].fillna('').astype(str)
    for tok in proc_tokens:
        df[f'has_{tok}'] = s.str.contains(tok, regex=False).astype('int8')
    df = df.drop(columns=['특정 시술 유형'])

    # [4] 배아 생성 이유 토큰 분해 (→ 원본 컬럼 drop)
    s2 = df['배아 생성 주요 이유'].fillna('').astype(str)
    for r in reasons:
        df[f'reason_{r}'] = s2.str.contains(r, regex=False).astype('int8')
    df = df.drop(columns=['배아 생성 주요 이유'])

    # [5] 기본 비율 7개 (모든 모델 공통)
    eps = 1e-6
    df['past_preg_rate']        = (df['총 임신 횟수']       / (df['총 시술 횟수']   + eps)).astype('float32')
    df['past_birth_rate']       = (df['총 출산 횟수']       / (df['총 임신 횟수']   + eps)).astype('float32')
    df['ivf_preg_rate']         = (df['IVF 임신 횟수']      / (df['IVF 시술 횟수'] + eps)).astype('float32')
    df['embryo_transfer_ratio'] = (df['이식된 배아 수']     / (df['총 생성 배아 수']+ eps)).astype('float32')
    df['embryo_stored_ratio']   = (df['저장된 배아 수']     / (df['총 생성 배아 수']+ eps)).astype('float32')
    df['icsi_ratio']            = (df['미세주입된 난자 수'] / (df['혼합된 난자 수'] + eps)).astype('float32')
    df['age_x_transferred']     = (df['age_num']            * df['이식된 배아 수']  ).astype('float32')
    df['gap_transfer_minus_pickup'] = (df['배아 이식 경과일'] - df['난자 채취 경과일']).astype('float32')

    # [6] 결측 플래그 3개
    for c in ['착상 전 유전 검사 사용 여부', '임신 시도 또는 마지막 임신 경과 연수', '배아 해동 경과일']:
        if c in df.columns:
            df[f'{c}__isnan'] = df[c].isna().astype('int8')

    return df


def add_v3_features(df: pd.DataFrame) -> pd.DataFrame:
    """v10 동일: v3 핵심 도메인 피처 5개"""
    df['days_mix_to_transfer'] = (
        df['배아 이식 경과일'] - df['난자 혼합 경과일']
    ).astype('float32')

    is_donor = (df['난자 출처'].astype(str) == '기증 제공')
    df['real_age_num'] = df['age_num'].astype('float32').values.copy()
    df.loc[is_donor, 'real_age_num'] = df.loc[is_donor, 'egg_donor_age_num']
    df['real_age_num'] = df['real_age_num'].astype('float32')

    df['log_n_embryo'] = np.log1p(df['총 생성 배아 수'].fillna(0)).astype('float32')

    df['miss_gene_test'] = (
        df['PGS 시술 여부'].isna() |
        df['PGD 시술 여부'].isna() |
        df['착상 전 유전 검사 사용 여부'].isna()
    ).astype('int8')

    df['age_x_embryo_ratio'] = (
        df['age_num'] * df['embryo_transfer_ratio']
    ).astype('float32')

    return df


def add_v11_features(df: pd.DataFrame) -> pd.DataFrame:
    """v10 피처 전체 + v11 신규 임상 피처"""
    df = add_v3_features(df)
    eps = 1e-6

    # ── v10 Group E: 결측 핵심 플래그 ─────────────────────────────
    df['미싱_배아이식경과일'] = df['배아 이식 경과일'].isna().astype('int8')
    df['미싱_난자채취경과일'] = df['난자 채취 경과일'].isna().astype('int8')
    df['미싱_난자혼합경과일'] = df['난자 혼합 경과일'].isna().astype('int8')
    df['n_missing_row']      = df.isna().sum(axis=1).astype('int16')

    # ── v10 Group A: 시술 이력 ─────────────────────────────────────
    df['ever_delivered']   = (df['총 출산 횟수'].fillna(0) > 0).astype('int8')
    df['repeated_3plus']   = (df['총 시술 횟수'].fillna(0) >= 3).astype('int8')
    df['is_first_attempt'] = (df['총 시술 횟수'].fillna(-1) == 0).astype('int8')
    df['prev_birth_rate']  = (df['총 출산 횟수'].fillna(0) / (df['총 임신 횟수'].fillna(0) + 1)).astype('float32')
    df['유산_횟수']         = (df['총 임신 횟수'].fillna(0) - df['총 출산 횟수'].fillna(0)).clip(lower=0).astype('float32')
    df['long_infertility'] = (df['임신 시도 또는 마지막 임신 경과 연수'].fillna(0) >= 7).astype('int8')
    df['censored_시술']    = (df['총 시술 횟수'] == 6).astype('int8')
    df['censored_임신']    = (df['총 임신 횟수'] == 6).astype('int8')

    # ── v10 Group B: 배아 파이프라인 ──────────────────────────────
    df['성숙률']         = (df['미세주입된 난자 수'].fillna(0) / (df['수집된 신선 난자 수'].fillna(0) + 1)).astype('float32')
    df['ICSI_수정률']    = (df['미세주입에서 생성된 배아 수'].fillna(0) / (df['미세주입된 난자 수'].fillna(0) + 1)).astype('float32')
    df['이식가능률']     = ((df['이식된 배아 수'].fillna(0) + df['저장된 배아 수'].fillna(0)) / (df['총 생성 배아 수'].fillna(0) + 1)).astype('float32')
    df['이식채택률']     = (df['이식된 배아 수'].fillna(0) / (df['이식된 배아 수'].fillna(0) + df['저장된 배아 수'].fillna(0) + 1)).astype('float32')
    df['is_ICSI']        = (df['미세주입된 난자 수'].fillna(0) > 0).astype('int8')
    df['ICSI_실패수']    = (df['미세주입된 난자 수'].fillna(0) - df['미세주입에서 생성된 배아 수'].fillna(0)).clip(lower=0).astype('float32')
    df['ICSI_비중']      = (df['미세주입된 난자 수'].fillna(0) / (df['총 생성 배아 수'].fillna(0) + 1)).astype('float32')
    df['이식_취소']      = (df['이식된 배아 수'].fillna(-1) == 0).astype('int8')

    # ── v10 Group C: 시술 적합성 ──────────────────────────────────
    male = (
        (df.get('남성 주 불임 원인', pd.Series(0, index=df.index)).fillna(0) == 1) |
        (df.get('남성 부 불임 원인', pd.Series(0, index=df.index)).fillna(0) == 1) |
        (df.get('불임 원인 - 남성 요인', pd.Series(0, index=df.index)).fillna(0) == 1)
    ).astype('int8')
    df['male_factor'] = male
    df['시술적합성'] = np.select(
        condlist=[
            (male == 1) & (df['is_ICSI'] == 1),
            (male == 0) & (df['is_ICSI'] == 1),
            (male == 1) & (df['is_ICSI'] == 0),
        ],
        choicelist=['AdequateICSI', 'OverusedICSI', 'UnderservedICSI'],
        default='Neither'
    ).astype(object)
    df['ICSI_적합'] = ((male == 1) & (df['is_ICSI'] == 1)).astype('int8')

    # ── v10 Group D: 구간 범주형 ──────────────────────────────────
    df['배아수_그룹'] = pd.cut(
        df['총 생성 배아 수'], bins=[-0.01, 6, 14, 20, 1000],
        labels=['Poor', 'Normal', 'High', 'Hyper']
    ).astype(object)
    df['ICSI_난자_구간'] = pd.cut(
        df['미세주입된 난자 수'], bins=[-0.01, 3, 9, 19, 1000],
        labels=['VeryLow', 'Low', 'Optimal', 'VeryHigh']
    ).astype(object)

    # ── v10 Group F: 상호작용 ─────────────────────────────────────
    df['age_x_procs']      = (df['age_num'].fillna(0) * df['총 시술 횟수'].fillna(0)).astype('float32')
    df['age_x_log_embryo'] = (df['age_num'].fillna(0) * df['log_n_embryo']).astype('float32')
    df['icsi_fert_x_male'] = (df['ICSI_수정률'] * df['male_factor']).astype('float32')

    # ── v10 Sanity Check ──────────────────────────────────────────
    embryo_fixed = np.maximum(
        df['총 생성 배아 수'].fillna(0).values,
        (df['이식된 배아 수'].fillna(0) + df['저장된 배아 수'].fillna(0)).values
    )
    df['총 생성 배아 수_fixed'] = embryo_fixed.astype('float32')
    icsi_emb_fixed = np.minimum(
        df['미세주입에서 생성된 배아 수'].fillna(0).values,
        df['미세주입된 난자 수'].fillna(0).values
    )
    df['미세주입에서 생성된 배아 수_fixed'] = icsi_emb_fixed.astype('float32')

    # ── v10 단계별 전환율 ─────────────────────────────────────────
    fresh_eggs   = df['수집된 신선 난자 수'].fillna(0).values
    icsi_attempt = df['미세주입된 난자 수'].fillna(0).values
    transferred  = df['이식된 배아 수'].fillna(0).values
    stored       = df['저장된 배아 수'].fillna(0).values

    df['egg_to_embryo_rate']        = (embryo_fixed    / (fresh_eggs    + 1)).astype('float32')
    df['icsi_success_rate']         = (icsi_emb_fixed  / (icsi_attempt  + 1)).astype('float32')
    df['embryo_utilization_rate']   = ((transferred + stored) / (embryo_fixed  + 1)).astype('float32')
    df['total_pipeline_efficiency'] = ((transferred + stored) / (fresh_eggs    + 1)).astype('float32')

    # ── v10 FET 분리 마스킹 ───────────────────────────────────────
    df['is_FET'] = (df['해동된 배아 수'].fillna(0) > 0).astype('int8')
    fet_mask = (df['is_FET'] == 1).values

    fresh_adj = df['수집된 신선 난자 수'].fillna(0).astype('float32').values.copy()
    fresh_adj[fet_mask] = -1.0
    df['신선난자_FET조정'] = fresh_adj.astype('float32')

    mixed_adj = df['혼합된 난자 수'].fillna(0).astype('float32').values.copy()
    mixed_adj[fet_mask] = -1.0
    df['혼합난자_FET조정'] = mixed_adj.astype('float32')

    # ── v10 정합성 체크 ───────────────────────────────────────────
    df['is_mismatch_icsi']   = (df['male_factor'] != df['is_ICSI']).astype('int8')
    df['is_split_strategy']  = ((df['혼합된 난자 수'].fillna(0) > 0) & (df['미세주입된 난자 수'].fillna(0) > 0)).astype('int8')
    df['is_donor_sperm']     = (df['기증자 정자와 혼합된 난자 수'].fillna(0) > 0).astype('int8')

    # ── v10 나이 정규화 효율 (나이/35 방향) ──────────────────────
    df['age_normalized_efficiency'] = (
        df['egg_to_embryo_rate'] * (df['real_age_num'].fillna(35) / 35)
    ).astype('float32')
    df['age_normalized_pipeline'] = (
        df['total_pipeline_efficiency'] * (df['real_age_num'].fillna(35) / 35)
    ).astype('float32')

    # ════════════════════════════════════════════════════════════
    # ⭐ v11 신규 임상 피처
    # ════════════════════════════════════════════════════════════

    # 1. 유산률 (유산 경험 = 자궁·면역 문제 시사)
    #    유산_횟수는 v10에 이미 있으므로, 비율과 플래그만 추가
    total_preg  = df['총 임신 횟수'].fillna(0)
    total_birth = df['총 출산 횟수'].fillna(0)
    misc_count  = (total_preg - total_birth).clip(lower=0)
    df['has_miscarriage']  = (misc_count > 0).astype('int8')
    df['miscarriage_rate'] = (misc_count / (total_preg + eps)).astype('float32')

    # 2. 불임 원인 복합도 (원인 개수 합산 → 복합일수록 성공률 감소)
    cause_cols = [c for c in [
        '불임 원인 - 난관 질환', '불임 원인 - 남성 요인', '불임 원인 - 배란 장애',
        '불임 원인 - 자궁내막증', '불임 원인 - 정자 농도', '불임 원인 - 정자 운동성',
        '불임 원인 - 정자 형태',
    ] if c in df.columns]
    if cause_cols:
        df['infertility_complexity'] = df[cause_cols].fillna(0).sum(axis=1).astype('int8')

    # 3. 선택적 단일 배아 이식 (배아가 많은데 하나만 선택 = 품질 자신감 신호)
    df['is_elective_SET'] = (
        (df['이식된 배아 수'].fillna(0) == 1) & (df['총 생성 배아 수'].fillna(0) > 1)
    ).astype('int8')

    # 4. 과자극 위험군 (난자 20개 초과 = OHSS 위험 / 배아 품질 저하 가능)
    df['high_ovarian_response'] = (df['수집된 신선 난자 수'].fillna(0) > 20).astype('int8')

    # 5. 경과일 기반 배반포 이식 여부 (텍스트 토큰과 독립적 교차 검증)
    df['is_blastocyst_by_day'] = (
        df['배아 이식 경과일'].fillna(-1) >= 5
    ).astype('int8')

    # 6. ICSI 수정률 우수 여부 (0.70 기준 — 평균 수정률 약 75%)
    df['icsi_above_avg_success'] = np.where(
        df['is_ICSI'] == 1,
        (df['ICSI_수정률'] >= 0.70).astype('int8'),
        0
    ).astype('int8')

    # 7. DI 시술 여부 (IVF와 생물학적 메커니즘 근본적으로 다름)
    df['is_DI'] = (df['시술 유형'].astype(str) == 'DI').astype('int8')

    # ════════════════════════════════════════════════════════════
    # ⭐ v11.1 추가 임상 피처 (논문 근거)
    # ════════════════════════════════════════════════════════════

    # 8. is_egg_optimal — Sunkara 2011: 난자 10~20개가 성공률 정점
    #    모델이 '15개 근처'의 특수성을 이해하도록 최적 구간을 명시
    #    (high_ovarian_response >20 과 쌍으로 사용 — 좌우 비선형 경계)
    df['is_egg_optimal'] = (
        df['수집된 신선 난자 수'].fillna(0).between(10, 20)
    ).astype('int8')

    # 9. proven_fertility_older — Templeton 1996 핵심 교호작용
    #    40세 이상이지만 과거 출산 경험 있음 = 성공 확률 비약적 상승
    #    단순히 나이와 출산 경험을 따로 두지 않고 시너지를 직접 주입
    df['proven_fertility_older'] = (
        (df['age_num'].fillna(0) >= 40) & (df['ever_delivered'] == 1)
    ).astype('int8')

    # 10. dor_young — 젊은 저반응 (DOR: Diminished Ovarian Reserve) 신호
    #     35세 미만인데 난자 3개 이하 → 예상 밖 저반응 = 강력한 부정 신호
    #     같은 난자 수라도 고령 환자와 완전히 다른 의미를 가짐
    df['dor_young'] = (
        (df['수집된 신선 난자 수'].fillna(0) <= 3) &
        (df['age_num'].fillna(99) < 35)
    ).astype('int8')

    # 11. elective_freeze — 선택적 동결 주기
    #     이식 0개 + 저장 1개 이상 = 의도적 전배아 동결 보존 전략
    #     (내막 수용성 불량 or 과자극 회복 or OHSS 예방 목적)
    #     이식_취소와 구별: 이식_취소는 배아 없음까지 포함, 이건 배아가 있는 경우만
    df['elective_freeze'] = (
        (df['이식된 배아 수'].fillna(0) == 0) &
        (df['저장된 배아 수'].fillna(0) > 0)
    ).astype('int8')

    # 12. pgt_elder — PGT 사용 + 고령 (유전 선별 전략 신호)
    #     40세 이상에서 PGS/PGD 사용 = 배아 염색체 선별 적극 시행
    #     선별 후 이식이므로 성공 기대값 상승 (단, 이식 가능 배아 감소 위험도 있음)
    df['pgt_elder'] = (
        (df['age_num'].fillna(0) >= 40) &
        (
            (df.get('PGS 시술 여부', pd.Series(0, index=df.index)).fillna(0) == 1) |
            (df.get('PGD 시술 여부', pd.Series(0, index=df.index)).fillna(0) == 1)
        )
    ).astype('int8')

    # 13. male_severe — 중증 남성인자 (정자 3대 지표 2개 이상 동시 이상)
    #     단순 남성인자(male_factor)보다 훨씬 강한 부정 신호
    #     정자 농도 + 운동성 + 형태 중 2개 이상 이상 = 복합 정자 장애
    sperm_issues = (
        df.get('불임 원인 - 정자 농도',  pd.Series(0, index=df.index)).fillna(0) +
        df.get('불임 원인 - 정자 운동성', pd.Series(0, index=df.index)).fillna(0) +
        df.get('불임 원인 - 정자 형태',   pd.Series(0, index=df.index)).fillna(0)
    )
    df['male_severe'] = (sperm_issues >= 2).astype('int8')

    # ════════════════════════════════════════════════════════════
    # ⭐ v11.2 판을 바꾸는 피처 — 실패 모드 명시 + 임상 사전확률
    # ════════════════════════════════════════════════════════════

    fresh_eggs  = df['수집된 신선 난자 수'].fillna(0)
    total_emb   = df['총 생성 배아 수'].fillna(0)
    transferred = df['이식된 배아 수'].fillna(0)
    stored_emb  = df['저장된 배아 수'].fillna(0)

    # ── [실패 모드 1] 전체 수정 실패 ─────────────────────────────
    # 난자는 채취했는데 배아가 단 하나도 형성되지 않음
    # → 난자 질 또는 정자 기능 완전 불량 → 임신 확률 거의 0%
    # 모델이 이 케이스를 다른 "적은 배아" 케이스와 구별해야 함
    df['total_fertilization_failure'] = (
        (fresh_eggs > 0) & (total_emb == 0)
    ).astype('int8')

    # ── [실패 모드 2] 전배아 발달 정지 ───────────────────────────
    # 배아는 형성됐는데 이식도 저장도 0 = 전부 발달 정지/퇴화
    # (elective_freeze와 구별: 이건 배아 자체가 살아남지 못한 것)
    df['all_embryos_arrested'] = (
        (total_emb > 0) &
        (transferred == 0) &
        (stored_emb == 0)
    ).astype('int8')

    # ── [실패 모드 3] 배아 손실 수 ───────────────────────────────
    # 총 생성 배아 - (이식 + 저장) = 중간에 사라진 배아
    # 손실이 클수록 배아 질이 나쁘다는 강력한 신호
    # (정합성 보정값 기반 계산으로 음수 방지)
    embryo_fixed = df['총 생성 배아 수_fixed'].fillna(total_emb)
    df['arrested_embryo_count'] = (
        embryo_fixed - transferred - stored_emb
    ).clip(lower=0).astype('float32')
    df['arrested_embryo_rate'] = (
        df['arrested_embryo_count'] / (embryo_fixed + eps)
    ).astype('float32')

    # ── [임상 사전확률] HFEA 나이대별 공표 성공률 ────────────────
    # UK HFEA가 수십만 건 데이터로 공표한 IVF 성공률 (2023 기준)
    # 단순 나이 ordinal보다 '보정된 사전확률'로 모델에 직접 주입
    # 기증 난자 사용 케이스는 기증자 나이 기준으로 별도 처리
    _hfea_prior = {0: 0.32, 1: 0.25, 2: 0.17, 3: 0.10, 4: 0.05, 5: 0.02}
    _donor_prior = 0.38  # 기증 난자 평균 성공률 (HFEA 2023)
    if 'age_ord' in df.columns:
        base_prior = df['age_ord'].map(_hfea_prior).fillna(0.17).astype('float32')
        is_donor = (df['난자 출처'].astype(str) == '기증 제공') if '난자 출처' in df.columns \
                   else pd.Series(False, index=df.index)
        base_prior[is_donor] = _donor_prior
        df['age_success_prior'] = base_prior

    # ── [기증 난자 이점] 나이 격차 → 난자 질 이점 ────────────────
    # 자신의 나이 - 기증자 나이 = 양수일수록 젊은 난자 사용 = 이점
    # (예: 본인 44세, 기증자 28세 → 이점 +16년)
    # 자가 난자 사용 시 0으로 처리
    if 'egg_donor_age_num' in df.columns:
        is_donor_mask = (df['난자 출처'].astype(str) == '기증 제공') if '난자 출처' in df.columns \
                        else pd.Series(False, index=df.index)
        df['donor_age_advantage'] = np.where(
            is_donor_mask,
            df['age_num'].fillna(36) - df['egg_donor_age_num'].fillna(28),
            0.0
        ).astype('float32')

    # ── [3관왕 조합] 최적 조건 동시 충족 ─────────────────────────
    # 40세 미만 + 배반포 이식 + 최적 난자 수(10~20개)
    # 세 조건이 동시에 충족될 때 성공률이 비선형적으로 급상승
    df['triple_positive'] = (
        (df['age_num'].fillna(99) < 40) &
        (df['배아 이식 경과일'].fillna(-1) >= 5) &
        (fresh_eggs.between(10, 20))
    ).astype('int8')

    # ── [배반포 + 선택적 단일 이식] 조합 시너지 ─────────────────
    # 배반포(Day5) × 선택적 SET = 가장 신뢰할 수 있는 임신 예측 조합
    # (has_BLASTOCYST 텍스트 토큰과 교차 검증)
    df['blast_eset'] = (
        (df['배아 이식 경과일'].fillna(-1) >= 5) &
        (df['is_elective_SET'] == 1)
    ).astype('int8')

    # ── [누적 실패 부담] ──────────────────────────────────────────
    # 총 시술 횟수 - 총 임신 횟수 = 지금까지 얼마나 실패했는가
    # 많을수록 심리적 소진 + 의학적으로 '난치성' 신호
    df['cumulative_failure_burden'] = (
        df['총 시술 횟수'].fillna(0) - df['총 임신 횟수'].fillna(0)
    ).clip(lower=0).astype('float32')

    # ── [클리닉 재방문 환자] ─────────────────────────────────────
    # 같은 클리닉에 3회 이상 = 클리닉이 이 환자 반응 패턴을 학습한 상태
    # 클리닉 맞춤 자극 프로토콜 최적화 가능 → 성공률 상승 가능성
    df['clinic_veteran'] = (
        df['클리닉 내 총 시술 횟수'].fillna(0) >= 3
    ).astype('int8')

    # ── [성숙 난자율] ─────────────────────────────────────────────
    # ICSI 시 미세주입 난자 / 수집 난자 = 성숙(MII) 난자 비율
    # 정상 범위: 70~80%. 이탈 시 난소 과자극 또는 자극 불충분 신호
    df['mature_oocyte_rate'] = (
        df['미세주입된 난자 수'].fillna(0) / (fresh_eggs + 1)
    ).astype('float32')

    return df


# =============================================================================
# 데이터 로드 (청크 단위)
# =============================================================================
def load_chunked(path: Path, chunksize: int = 20_000) -> pd.DataFrame:
    # dtype 사전 자동 생성
    probe = pd.read_csv(path, nrows=200)
    obj_cols = [c for c in probe.select_dtypes(include='object').columns if c != ID_COL]
    dtype_dict = {c: ('object' if c in obj_cols else 'float32') for c in probe.columns if c != ID_COL}
    del probe

    parts = []
    for chunk in pd.read_csv(path, dtype=dtype_dict, chunksize=chunksize):
        parts.append(process_chunk(chunk))
        gc.collect()
    df = pd.concat(parts, ignore_index=True)
    del parts; gc.collect()
    return df


# =============================================================================
# 범주형 처리 (Data Leakage 방지)
# =============================================================================
def make_categorical(train: pd.DataFrame, test: pd.DataFrame, cat_cols: list):
    cat_maps = {}
    for c in cat_cols:
        if c not in train.columns:
            continue
        train[c] = train[c].astype(str).replace({'nan': np.nan, 'None': np.nan})
        vals = sorted([v for v in train[c].dropna().unique()])
        cat_maps[c] = vals
        train[c] = pd.Categorical(train[c], categories=vals)
    for c in cat_cols:
        if c not in test.columns:
            continue
        test[c] = test[c].astype(str).replace({'nan': np.nan, 'None': np.nan})
        vals = cat_maps.get(c, [])
        unseen = (~test[c].isin(vals)) & test[c].notna()
        if unseen.any():
            test.loc[unseen, c] = np.nan
        test[c] = pd.Categorical(test[c], categories=vals)
    return train, test, cat_maps


# =============================================================================
# Target Encoding (Smoothed, Leakage-safe)
# =============================================================================
def _smoothed_te(values, y_vals, smoothing: float):
    gm  = float(np.mean(y_vals))
    agg = pd.DataFrame({'k': values, 'y': y_vals}).groupby('k', observed=True)['y'].agg(['mean','count'])
    sm  = (agg['mean'] * agg['count'] + gm * smoothing) / (agg['count'] + smoothing)
    return sm.to_dict(), gm


def build_te(train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray,
             te_cols: list, sm_map: dict, seed: int = 42) -> dict:
    """Leakage-safe Smoothed TE — train OOF / test fold 평균"""
    result = {}
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for col in te_cols:
        sm   = sm_map.get(col, 20.0)
        oof  = np.zeros(len(train), dtype=np.float32)
        t_pr = np.zeros(len(test),  dtype=np.float32)
        for _, (tri, vai) in enumerate(skf.split(train, y)):
            m, gm = _smoothed_te(train[col].iloc[tri].values, y[tri], sm)
            oof[vai] += train[col].iloc[vai].map(m).fillna(gm).values
            t_pr     += test[col].map(m).fillna(gm).values / N_FOLDS
        nm = f'TE__{col.lstrip("_te_")}'
        result[nm] = (oof, t_pr)
    return result


def add_te_combo_keys(df: pd.DataFrame) -> pd.DataFrame:
    """v10 동일: 3종 콤보 TE 키 생성"""
    df['_te_proc_specific'] = (
        df['has_ICSI'].astype(str) + df['has_IVF'].astype(str) +
        df['has_BLASTOCYST'].astype(str) + df['has_AH'].astype(str) +
        df['has_FER'].astype(str) + df['has_Unknown'].astype(str)
    )
    df['_te_proc_age']    = df['시술 유형'].astype(str) + '_' + df['시술 당시 나이'].astype(str)
    df['_te_proc_timing'] = df['시술 유형'].astype(str) + '_' + df['시술 시기 코드'].astype(str)
    return df


# =============================================================================
# 공통 K-Fold 학습 루프
# =============================================================================
def kfold_lgb(train, test, y, cat_cols, params, seeds, name, tracker=None):
    oof  = np.zeros(len(train), dtype=np.float32)
    pred = np.zeros(len(test),  dtype=np.float32)
    for sd in seeds:
        p   = {**params, 'random_state': sd, 'seed': sd, 'bagging_seed': sd, 'feature_fraction_seed': sd}
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=sd)
        os_ = np.zeros(len(train), dtype=np.float32)
        ps_ = np.zeros(len(test),  dtype=np.float32)
        for fold, (tri, vai) in enumerate(skf.split(train, y)):
            t0  = time.time()
            dtr = lgb.Dataset(train.iloc[tri], label=y[tri],   categorical_feature=cat_cols, free_raw_data=True)
            dva = lgb.Dataset(train.iloc[vai], label=y[vai],   categorical_feature=cat_cols, reference=dtr, free_raw_data=True)
            m   = lgb.train(p, dtr, num_boost_round=5000,
                            valid_sets=[dva], valid_names=['v'],
                            callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)])
            os_[vai] = m.predict(train.iloc[vai], num_iteration=m.best_iteration)
            ps_     += m.predict(test,            num_iteration=m.best_iteration) / N_FOLDS
            auc_f    = roc_auc_score(y[vai], os_[vai])
            if tracker: tracker.step(time.time()-t0, auc_f, sd, fold+1)
            del m, dtr, dva; gc.collect()
        seed_auc = roc_auc_score(y, os_)
        if tracker: tracker.seed_done(sd, seed_auc)
        oof += os_; pred += ps_
        del os_, ps_; gc.collect()
    oof  /= len(seeds)
    pred /= len(seeds)
    auc   = roc_auc_score(y, oof)
    if tracker: tracker.done(auc)
    return oof, pred, auc


def kfold_xgb(train, test, y, params, seeds, name, enable_cat=False, tracker=None):
    def _to_cat(df):
        if not enable_cat: return df
        df = df.copy()
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].astype('category')
        return df

    oof  = np.zeros(len(train), dtype=np.float32)
    pred = np.zeros(len(test),  dtype=np.float32)
    for sd in seeds:
        p   = {**params, 'seed': sd}
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=sd)
        os_ = np.zeros(len(train), dtype=np.float32)
        ps_ = np.zeros(len(test),  dtype=np.float32)
        for fold, (tri, vai) in enumerate(skf.split(train, y)):
            t0  = time.time()
            dtr = xgb.DMatrix(_to_cat(train.iloc[tri]), label=y[tri], enable_categorical=enable_cat)
            dva = xgb.DMatrix(_to_cat(train.iloc[vai]), label=y[vai], enable_categorical=enable_cat)
            dte = xgb.DMatrix(_to_cat(test),                          enable_categorical=enable_cat)
            m   = xgb.train(p, dtr, num_boost_round=3000,
                            evals=[(dva,'v')], early_stopping_rounds=100, verbose_eval=False)
            os_[vai] = m.predict(dva)
            ps_     += m.predict(dte) / N_FOLDS
            auc_f    = roc_auc_score(y[vai], os_[vai])
            if tracker: tracker.step(time.time()-t0, auc_f, sd, fold+1)
            del m, dtr, dva, dte; gc.collect()
        seed_auc = roc_auc_score(y, os_)
        if tracker: tracker.seed_done(sd, seed_auc)
        oof += os_; pred += ps_
        del os_, ps_; gc.collect()
    oof  /= len(seeds)
    pred /= len(seeds)
    auc   = roc_auc_score(y, oof)
    if tracker: tracker.done(auc)
    return oof, pred, auc


def kfold_cat(train, test, y, params, seeds, name, cat_idx, tracker=None):
    oof  = np.zeros(len(train), dtype=np.float32)
    pred = np.zeros(len(test),  dtype=np.float32)
    for sd in seeds:
        p   = {**params, 'random_seed': sd}
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=sd)
        os_ = np.zeros(len(train), dtype=np.float32)
        ps_ = np.zeros(len(test),  dtype=np.float32)
        for fold, (tri, vai) in enumerate(skf.split(train, y)):
            t0  = time.time()
            tp  = Pool(train.iloc[tri], y[tri], cat_features=cat_idx)
            vp  = Pool(train.iloc[vai], y[vai], cat_features=cat_idx)
            m   = CatBoostClassifier(**p)
            m.fit(tp, eval_set=vp, use_best_model=True, verbose=0)
            os_[vai] = m.predict_proba(train.iloc[vai])[:, 1]
            ps_     += m.predict_proba(test)[:, 1] / N_FOLDS
            auc_f    = roc_auc_score(y[vai], os_[vai])
            if tracker: tracker.step(time.time()-t0, auc_f, sd, fold+1)
            del m, tp, vp; gc.collect()
        seed_auc = roc_auc_score(y, os_)
        if tracker: tracker.seed_done(sd, seed_auc)
        oof += os_; pred += ps_
        del os_, ps_; gc.collect()
    oof  /= len(seeds)
    pred /= len(seeds)
    auc   = roc_auc_score(y, oof)
    if tracker: tracker.done(auc)
    return oof, pred, auc


# =============================================================================
# 분리 모델 (M5) — 이식 / 비이식 별도 LGB 학습
# =============================================================================
def run_split_model(train, test, y, cat_cols, seeds):
    transfer_col = '이식된 배아 수'
    if transfer_col not in train.columns:
        print('⚠️  이식된 배아 수 없음 — 분리 모델 스킵')
        return None, None, None

    tr_mask = (train[transfer_col].fillna(0) > 0).values
    te_mask = (test[transfer_col].fillna(0)  > 0).values
    print(f'\n[M5 분리 모델]  이식 {tr_mask.sum():,} / 비이식 {(~tr_mask).sum():,}')
    print(f'   이식 성공률   : {y[tr_mask].mean()*100:.2f}%')
    print(f'   비이식 성공률 : {y[~tr_mask].mean()*100:.2f}%')

    oof_all  = np.zeros(len(train), dtype=np.float32)
    pred_all = np.zeros(len(test),  dtype=np.float32)

    for gname, gtr, gte, mil in [
        ('NoTransfer', ~tr_mask, ~te_mask, 100),
        ('Transfer',    tr_mask,  te_mask, 200),
    ]:
        sub_tr  = train[gtr].reset_index(drop=True)
        sub_y   = y[gtr]
        sub_te  = test[gte].reset_index(drop=True)
        p       = {**LGB_PARAMS, 'min_data_in_leaf': mil}
        tracker = ProgressTracker(len(seeds) * N_FOLDS, f'M5 [{gname}]')

        oof_g  = np.zeros(len(sub_tr), dtype=np.float32)
        pred_g = np.zeros(len(sub_te), dtype=np.float32)

        for sd in seeds:
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=sd)
            pp  = {**p, 'random_state': sd, 'seed': sd, 'bagging_seed': sd, 'feature_fraction_seed': sd}
            os_ = np.zeros(len(sub_tr), dtype=np.float32)
            ps_ = np.zeros(len(sub_te), dtype=np.float32)

            for fold, (tri, vai) in enumerate(skf.split(sub_tr, sub_y)):
                t0  = time.time()
                dtr = lgb.Dataset(sub_tr.iloc[tri], label=sub_y[tri], categorical_feature=cat_cols, free_raw_data=True)
                dva = lgb.Dataset(sub_tr.iloc[vai], label=sub_y[vai], categorical_feature=cat_cols, reference=dtr, free_raw_data=True)
                m   = lgb.train(pp, dtr, num_boost_round=5000,
                                valid_sets=[dva], valid_names=['v'],
                                callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)])
                os_[vai] = m.predict(sub_tr.iloc[vai], num_iteration=m.best_iteration)
                ps_     += m.predict(sub_te,           num_iteration=m.best_iteration) / N_FOLDS
                auc_f    = roc_auc_score(sub_y[vai], os_[vai])
                tracker.step(time.time()-t0, auc_f, sd, fold+1)
                del m, dtr, dva; gc.collect()

            tracker.seed_done(sd, roc_auc_score(sub_y, os_))
            oof_g += os_; pred_g += ps_
            del os_, ps_; gc.collect()

        oof_g  /= len(seeds)
        pred_g /= len(seeds)
        tracker.done(roc_auc_score(sub_y, oof_g))
        oof_all[gtr]  = oof_g
        pred_all[gte] = pred_g

    auc = roc_auc_score(y, oof_all)
    return oof_all, pred_all, auc


# =============================================================================
# 앙상블 (Nelder-Mead, 5 시작점)
# =============================================================================
def run_ensemble(oofs: dict, preds: dict, y: np.ndarray):
    keys     = list(oofs.keys())
    oof_arr  = np.stack([oofs[k]  for k in keys], axis=1).astype(np.float64)
    pred_arr = np.stack([preds[k] for k in keys], axis=1).astype(np.float64)

    def neg_auc(w):
        w = np.clip(w, 0, None)
        s = w.sum()
        return 0.0 if s < 1e-9 else -roc_auc_score(y, oof_arr @ (w / s))

    n = len(keys)
    starts = [
        [0.0, 0.31, 0.165, 0.387, 0.138, 0.0][:n] + [1/n] * max(0, n-6),
        [0.0, 0.25, 0.15,  0.30,  0.10,  0.20][:n] + [1/n] * max(0, n-6),
        [0.0, 0.20, 0.15,  0.25,  0.10,  0.30][:n] + [1/n] * max(0, n-6),
        [0.1, 0.20, 0.15,  0.25,  0.10,  0.20][:n] + [1/n] * max(0, n-6),
        [1/n] * n,
    ]

    best_w, best_auc = None, -np.inf
    print(f'\n[앙상블]  {keys}')
    print('\n  개별 OOF AUC:')
    for k in keys:
        print(f'    {k:20s}  {roc_auc_score(y, oofs[k]):.5f}')

    for st in starts:
        st  = (list(st) + [1/n]*n)[:n]
        res = minimize(neg_auc, np.array(st), method='Nelder-Mead',
                       options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-7})
        w   = np.clip(res.x, 0, None); w /= w.sum()
        auc = roc_auc_score(y, oof_arr @ w)
        if auc > best_auc:
            best_auc, best_w = auc, w

    print('\n  최적 가중치:')
    for k, v in zip(keys, best_w):
        print(f'    {k:20s}  {v:.4f}')
    print(f'\n  앙상블 OOF AUC: {best_auc:.5f}  ★')
    return oof_arr @ best_w, pred_arr @ best_w, best_w


# =============================================================================
# 제출 파일 저장
# =============================================================================
def save_sub(pred, sub_template, auc, tag):
    sub  = sub_template.copy()
    col  = [c for c in sub.columns if c != ID_COL][0]
    sub[col] = np.clip(pred, 0, 1)
    ts   = datetime.now().strftime('%m%d_%H%M')
    path = OUT_DIR / f'submission_{tag}_AUC{auc:.5f}_{ts}.csv'
    sub.to_csv(path, index=False)
    print(f'  💾 {path}  (mean={sub[col].mean():.4f}  std={sub[col].std():.4f})')


# =============================================================================
# 메인
# =============================================================================
def main(clear_cache: bool = False):
    """
    clear_cache=True : output/*.npy 캐시 삭제 후 전체 재학습
    clear_cache=False: 캐시 있으면 로드, 없으면 학습 (기본값)
    """
    if clear_cache:
        cleared = list(OUT_DIR.glob('*.npy'))
        for f in cleared:
            f.unlink()
        print(f'🗑️  캐시 {len(cleared)}개 삭제 완료 — 전체 재학습합니다')

    pipeline = PipelineBar()

    # ── 0. 데이터 로드 ────────────────────────────────────────────
    assert (DATA_DIR/'train.csv').exists(), f'train.csv 없음: {DATA_DIR}'
    print('\n[1] 데이터 로드 중...')
    train_raw = load_chunked(DATA_DIR / 'train.csv')
    test_raw  = load_chunked(DATA_DIR / 'test.csv')
    sub_tmpl  = pd.read_csv(DATA_DIR / 'sample_submission.csv')

    y         = train_raw[TARGET].astype(np.int8).values
    train_ids = train_raw[ID_COL].values
    test_ids  = test_raw[ID_COL].values

    print(f'  train {train_raw.shape}  test {test_raw.shape}')
    print(f'  성공률: {y.mean():.4f}')

    pipeline.advance('완료')   # ⚙️  데이터 로드 + FE

    oofs, preds = {}, {}

    # ══════════════════════════════════════════════════════════════
    # M1. LightGBM v3  (SEEDS_5 × 5-Fold)
    # ══════════════════════════════════════════════════════════════
    cache_oof = OUT_DIR / 'oof_m1_lgb_v3.npy'
    cache_pred = OUT_DIR / 'pred_m1_lgb_v3.npy'
    if cache_oof.exists():
        oof_m1 = np.load(cache_oof); pred_m1 = np.load(cache_pred)
        print(f'[skip] M1 loaded  OOF={roc_auc_score(y, oof_m1):.5f}')
        auc_m1 = roc_auc_score(y, oof_m1)
    else:
        tr = train_raw.drop(columns=[TARGET, ID_COL]).copy()
        te = test_raw.drop(columns=[ID_COL]).copy()
        tr = add_v3_features(tr); te = add_v3_features(te)
        cat_v3 = [c for c in KEEP_CAT_COLS if c in tr.columns]
        tr, te, _ = make_categorical(tr, te, cat_v3)
        tracker = ProgressTracker(len(SEEDS_5)*N_FOLDS, 'M1 LGB v3')
        oof_m1, pred_m1, auc_m1 = kfold_lgb(tr, te, y, cat_v3, LGB_PARAMS, SEEDS_5, 'M1', tracker)
        np.save(cache_oof, oof_m1); np.save(cache_pred, pred_m1)
        del tr, te; gc.collect()
    oofs['LGB_v3'] = oof_m1; preds['LGB_v3'] = pred_m1
    pipeline.advance(f'OOF={auc_m1:.5f}')

    # ══════════════════════════════════════════════════════════════
    # M2. LightGBM v3 + Smoothed TE 3종  (SEEDS_5)
    # ══════════════════════════════════════════════════════════════
    cache_oof = OUT_DIR / 'oof_m2_lgb_v3te.npy'
    cache_pred = OUT_DIR / 'pred_m2_lgb_v3te.npy'
    if cache_oof.exists():
        oof_m2 = np.load(cache_oof); pred_m2 = np.load(cache_pred)
        print(f'[skip] M2 loaded  OOF={roc_auc_score(y, oof_m2):.5f}')
        auc_m2 = roc_auc_score(y, oof_m2)
    else:
        tr = train_raw.drop(columns=[TARGET, ID_COL]).copy()
        te = test_raw.drop(columns=[ID_COL]).copy()
        tr = add_v3_features(tr); te = add_v3_features(te)
        tr = add_te_combo_keys(tr); te = add_te_combo_keys(te)
        te_cols = ['_te_proc_specific', '_te_proc_age', '_te_proc_timing']
        sm_map  = {'_te_proc_specific': 30.0, '_te_proc_age': 25.0, '_te_proc_timing': 25.0}
        te_feats = build_te(tr, te, y, te_cols, sm_map)
        for nm, (oof_te, pred_te) in te_feats.items():
            tr[nm] = oof_te; te[nm] = pred_te
        tr = tr.drop(columns=te_cols); te = te.drop(columns=te_cols)
        cat_v3 = [c for c in KEEP_CAT_COLS if c in tr.columns]
        tr, te, _ = make_categorical(tr, te, cat_v3)
        tracker = ProgressTracker(len(SEEDS_5)*N_FOLDS, 'M2 LGB v3+TE')
        oof_m2, pred_m2, auc_m2 = kfold_lgb(tr, te, y, cat_v3, LGB_PARAMS, SEEDS_5, 'M2', tracker)
        np.save(cache_oof, oof_m2); np.save(cache_pred, pred_m2)
        del tr, te; gc.collect()
    oofs['LGB_v3TE'] = oof_m2; preds['LGB_v3TE'] = pred_m2
    pipeline.advance(f'OOF={auc_m2:.5f}')

    # ══════════════════════════════════════════════════════════════
    # M3. XGBoost v3feat  (SEEDS_3)
    # ══════════════════════════════════════════════════════════════
    cache_oof = OUT_DIR / 'oof_m3_xgb_v3.npy'
    cache_pred = OUT_DIR / 'pred_m3_xgb_v3.npy'
    if cache_oof.exists():
        oof_m3 = np.load(cache_oof); pred_m3 = np.load(cache_pred)
        print(f'[skip] M3 loaded  OOF={roc_auc_score(y, oof_m3):.5f}')
        auc_m3 = roc_auc_score(y, oof_m3)
    else:
        tr = train_raw.drop(columns=[TARGET, ID_COL]).copy()
        te = test_raw.drop(columns=[ID_COL]).copy()
        tr = add_v3_features(tr); te = add_v3_features(te)
        cat_v3 = [c for c in KEEP_CAT_COLS if c in tr.columns]
        tr, te, _ = make_categorical(tr, te, cat_v3)
        # float64 → float32
        for c in tr.select_dtypes(include='float64').columns:
            tr[c] = tr[c].astype('float32'); te[c] = te[c].astype('float32')
        tracker = ProgressTracker(len(SEEDS_3)*N_FOLDS, 'M3 XGB v3feat')
        oof_m3, pred_m3, auc_m3 = kfold_xgb(tr, te, y, XGB_PARAMS, SEEDS_3, 'M3',
                                              enable_cat=True, tracker=tracker)
        np.save(cache_oof, oof_m3); np.save(cache_pred, pred_m3)
        del tr, te; gc.collect()
    oofs['XGB_v3'] = oof_m3; preds['XGB_v3'] = pred_m3
    pipeline.advance(f'OOF={auc_m3:.5f}')

    # ══════════════════════════════════════════════════════════════
    # M4. XGBoost v11 (신규 피처 포함)  (SEEDS_3)
    # ══════════════════════════════════════════════════════════════
    cache_oof = OUT_DIR / 'oof_m4_xgb_v11.npy'
    cache_pred = OUT_DIR / 'pred_m4_xgb_v11.npy'
    if cache_oof.exists():
        oof_m4 = np.load(cache_oof); pred_m4 = np.load(cache_pred)
        print(f'[skip] M4 loaded  OOF={roc_auc_score(y, oof_m4):.5f}')
        auc_m4 = roc_auc_score(y, oof_m4)
    else:
        tr = train_raw.drop(columns=[TARGET, ID_COL]).copy()
        te = test_raw.drop(columns=[ID_COL]).copy()
        tr = add_v11_features(tr); te = add_v11_features(te)
        cat_v11 = [c for c in KEEP_CAT_COLS + KEEP_CAT_V11 if c in tr.columns]
        tr, te, _ = make_categorical(tr, te, cat_v11)
        for c in tr.select_dtypes(include='float64').columns:
            tr[c] = tr[c].astype('float32'); te[c] = te[c].astype('float32')
        print(f'  M4 피처 수: {len(tr.columns)}')
        tracker = ProgressTracker(len(SEEDS_3)*N_FOLDS, 'M4 XGB v11')
        oof_m4, pred_m4, auc_m4 = kfold_xgb(tr, te, y, XGB_PARAMS, SEEDS_3, 'M4',
                                              enable_cat=True, tracker=tracker)
        np.save(cache_oof, oof_m4); np.save(cache_pred, pred_m4)
        del tr, te; gc.collect()
    oofs['XGB_v11'] = oof_m4; preds['XGB_v11'] = pred_m4
    pipeline.advance(f'OOF={auc_m4:.5f}')

    # ══════════════════════════════════════════════════════════════
    # M5. 이식/비이식 분리 모델  (v3, SEEDS_5)
    # ══════════════════════════════════════════════════════════════
    cache_oof = OUT_DIR / 'oof_m5_split.npy'
    cache_pred = OUT_DIR / 'pred_m5_split.npy'
    if cache_oof.exists():
        oof_m5 = np.load(cache_oof); pred_m5 = np.load(cache_pred)
        print(f'[skip] M5 loaded  OOF={roc_auc_score(y, oof_m5):.5f}')
        auc_m5 = roc_auc_score(y, oof_m5)
    else:
        tr = train_raw.drop(columns=[TARGET, ID_COL]).copy()
        te = test_raw.drop(columns=[ID_COL]).copy()
        tr = add_v3_features(tr); te = add_v3_features(te)
        cat_v3 = [c for c in KEEP_CAT_COLS if c in tr.columns]
        tr, te, _ = make_categorical(tr, te, cat_v3)
        oof_m5, pred_m5, auc_m5 = run_split_model(tr, te, y, cat_v3, SEEDS_5)
        if oof_m5 is not None:
            np.save(cache_oof, oof_m5); np.save(cache_pred, pred_m5)
        del tr, te; gc.collect()
    if oof_m5 is not None:
        oofs['split'] = oof_m5; preds['split'] = pred_m5
    pipeline.advance(f'OOF={auc_m5:.5f}' if oof_m5 is not None else '스킵')

    # ══════════════════════════════════════════════════════════════
    # M6. CatBoost v11  (SEEDS_3)
    # ══════════════════════════════════════════════════════════════
    cache_oof = OUT_DIR / 'oof_m6_catboost.npy'
    cache_pred = OUT_DIR / 'pred_m6_catboost.npy'
    if cache_oof.exists():
        oof_m6 = np.load(cache_oof); pred_m6 = np.load(cache_pred)
        print(f'[skip] M6 loaded  OOF={roc_auc_score(y, oof_m6):.5f}')
        auc_m6 = roc_auc_score(y, oof_m6)
    else:
        tr = train_raw.drop(columns=[TARGET, ID_COL]).copy()
        te = test_raw.drop(columns=[ID_COL]).copy()
        tr = add_v11_features(tr); te = add_v11_features(te)
        cat_v11 = [c for c in KEEP_CAT_COLS + KEEP_CAT_V11 if c in tr.columns]
        # CatBoost: string 유지
        for c in cat_v11:
            if c in tr.columns:
                tr[c] = tr[c].astype(object).where(tr[c].notna(), 'NaN').astype(str)
                te[c] = te[c].astype(object).where(te[c].notna(), 'NaN').astype(str)
        for c in tr.select_dtypes(include='float64').columns:
            tr[c] = tr[c].astype('float32'); te[c] = te[c].astype('float32')
        feats   = tr.columns.tolist()
        cat_idx = [feats.index(c) for c in cat_v11 if c in feats]
        print(f'  M6 피처 수: {len(feats)}  cat: {len(cat_idx)}')
        tracker = ProgressTracker(len(SEEDS_3)*N_FOLDS, 'M6 CatBoost v11')
        oof_m6, pred_m6, auc_m6 = kfold_cat(tr, te, y, CAT_PARAMS, SEEDS_3, 'M6', cat_idx, tracker)
        np.save(cache_oof, oof_m6); np.save(cache_pred, pred_m6)
        del tr, te; gc.collect()
    oofs['CatBoost'] = oof_m6; preds['CatBoost'] = pred_m6
    pipeline.advance(f'OOF={auc_m6:.5f}')

    # ══════════════════════════════════════════════════════════════
    # 앙상블 (Nelder-Mead)
    # ══════════════════════════════════════════════════════════════
    oof_ens, pred_ens, weights = run_ensemble(oofs, preds, y)
    ens_auc = roc_auc_score(y, oof_ens)
    np.save(OUT_DIR / 'oof_ensemble.npy',  oof_ens)
    np.save(OUT_DIR / 'pred_ensemble.npy', pred_ens)
    save_sub(pred_ens, sub_tmpl, ens_auc, 'v11_ensemble')
    pipeline.advance(f'앙상블 AUC={ens_auc:.5f}')

    pipeline.finish(best_auc=ens_auc)
    return oofs, preds, oof_ens, pred_ens


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--clear-cache', action='store_true',
                        help='output/*.npy 캐시 삭제 후 전체 재학습')
    args = parser.parse_args()
    main(clear_cache=args.clear_cache)