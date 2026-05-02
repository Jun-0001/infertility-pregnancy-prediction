import pandas as pd
import numpy as np
import re
from pipeline.schema import (
    DROP_COLUMNS, CENSORED_COLUMNS, CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS, CENSORED_FLAG_COLUMNS,
)


class IVFPreprocessor:
    def __init__(self):
        self.medians = {}
        self.is_fitted = False

    # ─────────────────────────────────────────────────────────────────
    # 헬퍼
    # ─────────────────────────────────────────────────────────────────
    def _clean_string_to_int(self, val):
        """'0회'~'6회 이상' 문자열 → 정수"""
        if pd.isna(val) or val == '알 수 없음':
            return np.nan
        m = re.search(r'\d+', str(val))
        return int(m.group()) if m else np.nan

    # ─────────────────────────────────────────────────────────────────
    # fit / transform
    # ─────────────────────────────────────────────────────────────────
    def fit(self, df):
        """Train 통계량 학습 (누수 방지 — train only)"""
        for col in NUMERICAL_COLUMNS:
            if col in df.columns:
                self.medians[col] = pd.to_numeric(df[col], errors='coerce').median()
        self.is_fitted = True
        return self

    def transform(self, df):
        if not self.is_fitted:
            raise ValueError('Preprocessor를 먼저 fit 해주세요.')
        curr_df = df.copy()

        # ── Step 1. DROP 무의미 컬럼 ─────────────────────────────────
        curr_df = curr_df.drop(
            columns=[c for c in DROP_COLUMNS if c in curr_df.columns]
        )

        # ── Step 2. Censored 처리 — 플래그 생성 후 숫자 변환 ────────
        censored_flag_map = {
            '총 시술 횟수':           'is_censored_total_cycles',
            '클리닉 내 총 시술 횟수': 'is_censored_clinic_cycles',
            'IVF 시술 횟수':          'is_censored_prev_ivf',
            '총 임신 횟수':           'is_censored_prev_preg',
            '총 출산 횟수':           'is_censored_prev_birth',
        }
        for col, flag in censored_flag_map.items():
            if col in curr_df.columns:
                curr_df[flag] = (
                    curr_df[col].astype(str).str.contains('이상|>', regex=False)
                ).astype('int8')

        for col in CENSORED_COLUMNS:
            if col in curr_df.columns:
                curr_df[col] = pd.to_numeric(
                    curr_df[col].apply(self._clean_string_to_int), errors='coerce'
                )

        # ── Step 3. 나이 → Ordinal / Median ─────────────────────────
        # ★ '알 수 없음': np.nan (이전 -1 버그 수정)
        #    트리 모델은 NaN을 자체 처리. -1은 "가장 어린 나이보다 어리다"로 오해 유발
        age_ord_map = {
            '만18-34세': 0, '만35-37세': 1, '만38-39세': 2,
            '만40-42세': 3, '만43-44세': 4, '만45-50세': 5,
            '알 수 없음': np.nan,  # ★ 수정: -1 → NaN
        }
        age_median_map = {
            '만18-34세': 26.0, '만35-37세': 36.0, '만38-39세': 38.5,
            '만40-42세': 41.0, '만43-44세': 43.5, '만45-50세': 47.5,
            '알 수 없음': 36.0,
        }
        if '시술 당시 나이' in curr_df.columns:
            curr_df['age_ord']    = curr_df['시술 당시 나이'].map(age_ord_map)
            curr_df['age_median'] = curr_df['시술 당시 나이'].map(age_median_map).fillna(36.0)

        # ── Step 4. DI 시술 플래그 (구조적 분류) ────────────────────
        # 임상 의미: DI는 정자 기증 인공수정. IVF와 생물학적 메커니즘 완전히 다름.
        # DI 환자는 배아 관련 컬럼이 구조적으로 0/NaN → is_DI로 명시
        if '시술 유형' in curr_df.columns:
            curr_df['is_DI'] = (curr_df['시술 유형'] == 'DI').astype('int8')

       

        # ── Step 5. DI 환자 배아 관련 구조적 결측 → 0 ──────────────
        embryo_cols = [
            '총 생성 배아 수', '이식된 배아 수', '저장된 배아 수',
            '미세주입에서 생성된 배아 수', '미세주입 배아 이식 수', '미세주입 배아 저장 수',
        ]
        if '시술 유형' in curr_df.columns:
            di_mask = curr_df['시술 유형'] == 'DI'
            for col in embryo_cols:
                if col in curr_df.columns:
                    curr_df.loc[di_mask, col] = curr_df.loc[di_mask, col].fillna(0)

        # ── Step 6. 이식 경과일 Sentinel(-1) 처리 ────────────────────
        # ★ v10 반영: Sentinel 적용 전에 원본 결측 여부를 플래그로 보존
        #   (Sentinel -1이 들어가면 isna()로 잡을 수 없어짐)
        for raw_col, flag in [
            ('배아 이식 경과일', '미싱_배아이식경과일_raw'),
            ('난자 채취 경과일', '미싱_난자채취경과일_raw'),
            ('난자 혼합 경과일', '미싱_난자혼합경과일_raw'),
        ]:
            if raw_col in curr_df.columns:
                curr_df[flag] = curr_df[raw_col].isna().astype('int8')

        if '이식된 배아 수' in curr_df.columns and '배아 이식 경과일' in curr_df.columns:
            no_transfer = curr_df['이식된 배아 수'].fillna(0).eq(0)
            curr_df['no_transfer'] = no_transfer.astype('int8')
            curr_df.loc[no_transfer, '배아 이식 경과일'] = -1

        # ── Step 7. 해동/혼합 경과일 Sentinel(-1) 처리 ───────────────
        sentinel_pairs = {
            '해동된 배아 수': '배아 해동 경과일',
            '해동 난자 수':   '난자 해동 경과일',
            '혼합된 난자 수': '난자 혼합 경과일',
        }
        for cnt_col, day_col in sentinel_pairs.items():
            if cnt_col in curr_df.columns and day_col in curr_df.columns:
                zero_mask = curr_df[cnt_col].fillna(0).eq(0)
                curr_df.loc[zero_mask, day_col] = -1

        # ── Step 8. 유전검사 Flag NaN → 0 ────────────────────────────
        for col in ['착상 전 유전 검사 사용 여부', 'PGD 시술 여부', 'PGS 시술 여부']:
            if col in curr_df.columns:
                curr_df[col] = curr_df[col].fillna(0).astype('int8')

        # ── Step 9. 배아 생성 이유 Multi-hot 분해 ───────────────────
        reasons = ['현재 시술용', '기증용', '배아 저장용', '난자 저장용', '연구용']
        if '배아 생성 주요 이유' in curr_df.columns:
            for r in reasons:
                curr_df[f'is_{r}'] = curr_df['배아 생성 주요 이유'].fillna('').apply(
                    lambda x: 1 if r in str(x) else 0
                ).astype('int8')

        # ── Step 10. 수치형 결측 — Train 중앙값으로 보완 ─────────────
        # Sentinel(-1) 은 fillna 영향 없음 (NaN만 대상)
        for col, med in self.medians.items():
            if col in curr_df.columns:
                curr_df[col] = curr_df[col].fillna(med)

        return curr_df

    def fit_transform(self, df):
        return self.fit(df).transform(df)


if __name__ == '__main__':
    preprocessor = IVFPreprocessor()
    print('✅ IVFPreprocessor 준비 완료')