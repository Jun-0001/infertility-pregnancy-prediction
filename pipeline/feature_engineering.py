"""
feature_engineering.py — IVF 난임 예측 피처 엔지니어링

level 파라미터:
  'v3'  : _v3_domain만 실행 (앙상블 다양성, 모든 모델 공통 기반)
  'full': _v3_domain + Groups A~D 전체 (기본값)
"""

import pandas as pd
import numpy as np

_DONOR_AGE_MAP = {
    '만20세 이하': 19, '만21-25세': 23, '만26-30세': 28,
    '만31-35세': 33,   '만36-40세': 38, '만41-45세': 43,
    '알 수 없음': np.nan,
}

_PROC_TOKENS = ['ICSI', 'IVF', 'BLASTOCYST', 'AH', 'Unknown', 'FER']


class IVFFeatureEngineer:
    def __init__(self):
        pass

    def transform(self, df, level='full'):
        curr_df = df.copy()
        curr_df = self._v3_domain(curr_df)

        if level == 'v3':
            return curr_df

        def safe_num(col, fill=0.0):
            return pd.to_numeric(curr_df[col], errors='coerce').fillna(fill)

        # ─── Group A: 문헌 검증 킬러 피처 ────────────────────────────
        if '총 출산 횟수' in curr_df.columns:
            curr_df['ever_delivered'] = (safe_num('총 출산 횟수') > 0).astype('int8')

        if '총 시술 횟수' in curr_df.columns:
            proc = safe_num('총 시술 횟수')
            curr_df['repeated_3plus']   = (proc >= 3).astype('int8')
            curr_df['is_first_attempt'] = (proc == 0).astype('int8')

        if '해동된 배아 수' in curr_df.columns:
            curr_df['is_FET'] = (safe_num('해동된 배아 수') > 0).astype('int8')

        if '단일 배아 이식 여부' in curr_df.columns and '배아 이식 경과일' in curr_df.columns:
            curr_df['eset_blast'] = (
                (safe_num('단일 배아 이식 여부') == 1) &
                (safe_num('배아 이식 경과일', fill=-1) >= 5)
            ).astype('int8')

        if 'age_ord' in curr_df.columns and '난자 출처' in curr_df.columns:
            curr_df['elderly_donor_egg'] = (
                (safe_num('age_ord', fill=-1) >= 3) &
                (curr_df['난자 출처'].astype(str) == '기증 제공')
            ).astype('int8')

        # ─── Group B: 배아 파이프라인 전환율 ─────────────────────────
        fresh_eggs  = safe_num('수집된 신선 난자 수')
        total_emb   = safe_num('총 생성 배아 수')
        icsi_eggs   = safe_num('미세주입된 난자 수')
        icsi_emb    = safe_num('미세주입에서 생성된 배아 수')
        transferred = safe_num('이식된 배아 수')
        stored      = safe_num('저장된 배아 수')

        curr_df['ratio_maturity']         = icsi_eggs / (fresh_eggs + 1)
        curr_df['is_ICSI']                = (icsi_eggs > 0).astype('int8')
        curr_df['ratio_icsi_success']     = np.where(
            curr_df['is_ICSI'] == 1,
            icsi_emb / (icsi_eggs + 1),
            np.nan
        )
        curr_df['ratio_embryo_yield']     = total_emb / (fresh_eggs + 1)
        curr_df['ratio_transfer_density'] = transferred / (total_emb + 1)

        # ─── Group C: 도메인 상호작용 ────────────────────────────────
        if '수집된 신선 난자 수' in curr_df.columns:
            curr_df['eggs_optimal']          = fresh_eggs.between(10, 20).astype('int8')
            curr_df['low_ovarian_response']  = (fresh_eggs <= 3).astype('int8')
            # 신규: 과자극 위험군 (OHSS, >20개)
            curr_df['high_ovarian_response'] = (fresh_eggs > 20).astype('int8')

        if '불임 원인 - 남성 요인' in curr_df.columns:
            male = safe_num('불임 원인 - 남성 요인')
            curr_df['proper_icsi_match'] = ((male == 1) & (icsi_eggs > 0)).astype('int8')

        if 'age_median' in curr_df.columns and '총 시술 횟수' in curr_df.columns:
            curr_df['age_x_procedure'] = (
                safe_num('age_median', fill=36.0) * safe_num('총 시술 횟수')
            )

        # 신규: ICSI 수정률 우수 여부 (70% 기준)
        if 'ratio_icsi_success' in curr_df.columns:
            curr_df['icsi_above_avg_success'] = np.where(
                curr_df['is_ICSI'] == 1,
                (curr_df['ratio_icsi_success'] >= 0.70).astype('int8'),
                np.nan
            )

        # 신규: 경과일 기반 배반포 이식 확인 (has_BLASTOCYST 텍스트와 독립 검증)
        if '배아 이식 경과일' in curr_df.columns:
            curr_df['is_blastocyst_by_day'] = (
                safe_num('배아 이식 경과일', fill=-1) >= 5
            ).astype('int8')

        # ─── Group D: 구간/복합 피처 ─────────────────────────────────
        if '총 생성 배아 수' in curr_df.columns:
            curr_df['배아수_그룹'] = pd.cut(
                total_emb, bins=[-0.1, 6, 14, 20, 9999],
                labels=['Poor', 'Normal', 'High', 'Hyper']
            ).astype(str)

        if '미세주입된 난자 수' in curr_df.columns:
            curr_df['ICSI_난자_구간'] = pd.cut(
                icsi_eggs, bins=[-0.1, 3, 9, 19, 9999],
                labels=['VeryLow', 'Low', 'Optimal', 'VeryHigh']
            ).astype(str)

        if '불임 원인 - 남성 요인' in curr_df.columns:
            male_flag = safe_num('불임 원인 - 남성 요인').astype(int)
            icsi_flag = curr_df['is_ICSI']
            curr_df['시술적합성'] = np.select(
                condlist=[
                    (male_flag == 1) & (icsi_flag == 1),
                    (male_flag == 0) & (icsi_flag == 1),
                    (male_flag == 1) & (icsi_flag == 0),
                ],
                choicelist=['적합ICSI', '과사용', '부적합'],
                default='해당없음'
            )

        if '이식된 배아 수' in curr_df.columns:
            curr_df['이식_취소'] = (transferred == 0).astype('int8')

        is_fet_flag = curr_df.get('is_FET', pd.Series(0, index=curr_df.index))
        curr_df['flag_logic_error'] = (
            (~is_fet_flag.astype(bool)) & (transferred > total_emb)
        ).astype('int8')

        return curr_df

    # ─────────────────────────────────────────────────────────────────
    # _v3_domain: 모든 모델 공통 기반 피처
    # ─────────────────────────────────────────────────────────────────
    def _v3_domain(self, df):
        eps = 1e-6

        def to_f(col, fill=0.0):
            return (
                pd.to_numeric(df[col], errors='coerce').fillna(fill)
                if col in df.columns
                else pd.Series(fill, index=df.index, dtype='float32')
            )

        # 1. 혼합일 → 이식일 간격 (Day5 배반포 vs Day3 분할기 구분)
        if '배아 이식 경과일' in df.columns and '난자 혼합 경과일' in df.columns:
            df['days_mix_to_transfer'] = (
                pd.to_numeric(df['배아 이식 경과일'], errors='coerce') -
                pd.to_numeric(df['난자 혼합 경과일'],  errors='coerce')
            ).astype('float32')

        # 2. real_age_num: 기증 난자 사용 시 기증자 나이로 교체
        if 'age_median' in df.columns:
            real_age = pd.to_numeric(df['age_median'], errors='coerce').copy()
            if '난자 출처' in df.columns and '난자 기증자 나이' in df.columns:
                is_donor = df['난자 출처'].astype(str) == '기증 제공'
                real_age[is_donor] = df['난자 기증자 나이'].map(_DONOR_AGE_MAP)[is_donor]
            df['real_age_num'] = real_age.astype('float32')

        # 3. log_n_embryo
        if '총 생성 배아 수' in df.columns:
            df['log_n_embryo'] = np.log1p(
                pd.to_numeric(df['총 생성 배아 수'], errors='coerce').fillna(0)
            ).astype('float32')

        # 4. miss_gene_test (PGS/PGD/유전검사 결측 여부)
        gene_cols = ['PGS 시술 여부', 'PGD 시술 여부', '착상 전 유전 검사 사용 여부']
        present = [c for c in gene_cols if c in df.columns]
        if present:
            df['miss_gene_test'] = df[present].isna().any(axis=1).astype('int8')

        # 5. age_x_embryo_ratio (나이 × 이식 배아 비율)
        if 'age_median' in df.columns:
            age_val   = pd.to_numeric(df['age_median'], errors='coerce').fillna(36.0)
            emb_ratio = to_f('이식된 배아 수') / (to_f('총 생성 배아 수') + eps)
            df['age_x_embryo_ratio'] = (age_val * emb_ratio).astype('float32')

        # 6. proc 토큰 6개 (has_BLASTOCYST: 성공률 36.1% vs 평균 25.8%)
        if '특정 시술 유형' in df.columns:
            s = df['특정 시술 유형'].fillna('').astype(str)
            for tok in _PROC_TOKENS:
                df[f'has_{tok}'] = s.str.contains(tok, regex=False).astype('int8')

        # 7. v10 기본 비율 7개
        total_proc  = to_f('총 시술 횟수')
        total_preg  = to_f('총 임신 횟수')
        total_birth = to_f('총 출산 횟수')
        ivf_preg    = to_f('IVF 임신 횟수')
        ivf_proc    = to_f('IVF 시술 횟수')
        icsi_eggs   = to_f('미세주입된 난자 수')
        mixed_eggs  = to_f('혼합된 난자 수')
        transferred = to_f('이식된 배아 수')
        stored      = to_f('저장된 배아 수')
        total_emb   = to_f('총 생성 배아 수')
        age_val     = pd.to_numeric(
            df.get('age_median', pd.Series(36.0, index=df.index)),
            errors='coerce'
        ).fillna(36.0)

        df['past_preg_rate']        = (total_preg  / (total_proc  + eps)).astype('float32')
        df['past_birth_rate']       = (total_birth / (total_preg  + eps)).astype('float32')
        df['ivf_preg_rate']         = (ivf_preg    / (ivf_proc    + eps)).astype('float32')
        df['embryo_transfer_ratio'] = (transferred / (total_emb   + eps)).astype('float32')
        df['embryo_stored_ratio']   = (stored      / (total_emb   + eps)).astype('float32')
        df['icsi_ratio']            = (icsi_eggs   / (mixed_eggs  + eps)).astype('float32')
        df['age_x_transferred']     = (age_val     * transferred  ).astype('float32')

        # 8. age_normalized_efficiency — v10 동일 방향: 나이/35
        # "나이가 많은데도 효율이 높다" = 더 강력한 긍정 신호
        if 'real_age_num' in df.columns:
            age_clip   = df['real_age_num'].fillna(35.0).clip(lower=18.0)
            fresh_eggs = to_f('수집된 신선 난자 수')
            egg2emb    = total_emb / (fresh_eggs + 1)
            df['age_normalized_efficiency'] = (egg2emb * (age_clip / 35.0)).astype('float32')
            # 추가: 전체 파이프라인 효율도 나이 보정 (v10 age_normalized_pipeline)
            total_used = to_f('이식된 배아 수') + to_f('저장된 배아 수')
            pipeline_eff = total_used / (fresh_eggs + 1)
            df['age_normalized_pipeline'] = (pipeline_eff * (age_clip / 35.0)).astype('float32')

        # 9. is_elective_SET (선택적 단일 이식: 성공률 38.0% vs 11.1%)
        df['is_elective_SET'] = ((transferred == 1) & (total_emb > 1)).astype('int8')

        # 10. ★ 신규: 유산 경험 (반복 유산 = 자궁/면역 문제 시사)
        df['miscarriage_count'] = (total_preg - total_birth).clip(lower=0).astype('float32')
        df['has_miscarriage']   = (df['miscarriage_count'] > 0).astype('int8')
        df['miscarriage_rate']  = (df['miscarriage_count'] / (total_preg + eps)).astype('float32')

        # 11. ★ 신규: 불임 원인 복합도 (원인 수 → 성공률 감소)
        cause_cols = [
            c for c in [
                '불임 원인 - 난관 질환', '불임 원인 - 남성 요인',
                '불임 원인 - 배란 장애', '불임 원인 - 자궁내막증',
                '불임 원인 - 정자 농도', '불임 원인 - 정자 운동성',
                '불임 원인 - 정자 형태',
            ] if c in df.columns
        ]
        if cause_cols:
            df['infertility_complexity'] = (
                df[cause_cols].fillna(0).sum(axis=1).astype('int8')
            )

        # 12. ★ 신규: is_DI (DI 시술 vs IVF — 생물학적 메커니즘 완전히 다름)
        if 'is_DI' not in df.columns and '시술 유형' in df.columns:
            df['is_DI'] = (df['시술 유형'].astype(str) == 'DI').astype('int8')

        # 13. ★ v10 반영: 주요 경과일 결측 플래그 (구조적 결측 vs 진짜 결측 구분)
        for raw_col, flag_name in [
            ('배아 이식 경과일', '미싱_배아이식경과일'),
            ('난자 채취 경과일', '미싱_난자채취경과일'),
            ('난자 혼합 경과일', '미싱_난자혼합경과일'),
        ]:
            if raw_col in df.columns:
                df[flag_name] = df[raw_col].isna().astype('int8')
        df['n_missing_row'] = df.isna().sum(axis=1).astype('int16')

        # 14. ★ v10 반영: 장기 난임 플래그 (7년 이상 = 난치성 신호)
        if '임신 시도 또는 마지막 임신 경과 연수' in df.columns:
            df['long_infertility'] = (
                pd.to_numeric(df['임신 시도 또는 마지막 임신 경과 연수'], errors='coerce').fillna(0) >= 7
            ).astype('int8')

        # 15. ★ v10 반영: censored 플래그 (6회 이상 = 병력이 매우 긴 고위험군)
        if '총 시술 횟수' in df.columns:
            df['censored_시술'] = (to_f('총 시술 횟수') == 6).astype('int8')
        if '총 임신 횟수' in df.columns:
            df['censored_임신'] = (to_f('총 임신 횟수') == 6).astype('int8')

        return df


if __name__ == '__main__':
    eng = IVFFeatureEngineer()
    print('IVFFeatureEngineer 정의 완료')
    print()
    print('[_v3_domain — v3/full 공통]')
    print('  [복원] days_mix_to_transfer, real_age_num, log_n_embryo')
    print('  [복원] miss_gene_test, age_x_embryo_ratio')
    print('  [복원] ivf_preg_rate, embryo_transfer_ratio, embryo_stored_ratio')
    print('  [복원] icsi_ratio, age_x_transferred')
    print('  [유지] proc 6개, v10 비율 7개, age_normalized_efficiency, is_elective_SET')
    print('  [신규] miscarriage_count/rate/has_miscarriage')
    print('  [신규] infertility_complexity')
    print('  [신규] is_DI')
    print()
    print('[Groups A~D — full 전용]')
    print('  [복원] ratio_maturity, is_ICSI, ratio_icsi_success, ratio_embryo_yield')
    print('  [복원] ratio_transfer_density, low_ovarian_response, age_x_procedure')
    print('  [복원] 배아수_그룹, ICSI_난자_구간, 시술적합성, 이식_취소')
    print('  [신규] high_ovarian_response, is_blastocyst_by_day, icsi_above_avg_success')