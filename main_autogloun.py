import pandas as pd
from autogluon.tabular import TabularPredictor
from pipeline.schema import TARGET
from pipeline.preprocess import IVFPreprocessor
from pipeline.feature_engineering import IVFFeatureEngineer

def main():
    # 1. 데이터 로드
    train_raw = pd.read_csv('data/train.csv')
    test_raw = pd.read_csv('data/test.csv')
    
    # 2. 기존 인프라 활용 (재료 준비)
    pre = IVFPreprocessor().fit(train_raw)
    X_train_pre = pre.transform(train_raw)
    X_test_pre = pre.transform(test_raw)
    
    # 3. 킬러 피처 생성 (Full level 적용)
    fe = IVFFeatureEngineer()
    train_df = fe.transform(X_train_pre, level='full')
    test_df = fe.transform(X_test_pre, level='full')
    
    # 타겟 추가
    train_df[TARGET] = train_raw[TARGET]

    # 4. AutoGluon 가동 (모델링 & 앙상블 외주)
    # ※ 주의: AutoGluon은 범주형을 직접 처리하므로 
    #   기존의 finalize(LabelEncoding)를 거치지 않은 '문자열' 상태가 더 좋습니다.
    
    predictor = TabularPredictor(
        label=TARGET, 
        eval_metric='roc_auc',
        path='autogluon_ivf_model'
    ).fit(
        train_data=train_df,
        presets='best_quality',  # 멀티레이어 스태킹 활성화 (0.742 탈환 핵심)
        time_limit=3600 * 4      # 충분한 학습 시간 (4시간 권장)
    )

    # 5. 예측 및 제출 파일 생성
    predictions = predictor.predict_proba(test_df)
    
    sub = pd.read_csv('data/sample_submission.csv')
    sub[TARGET] = predictions.iloc[:, 1] # 성공(1) 확률값
    sub.to_csv('submission_autogluon.csv', index=False)

if __name__ == "__main__":
    main()