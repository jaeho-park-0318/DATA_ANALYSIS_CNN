"""기존 모델의 실패 유형과 장비·공정 분포를 읽기 전용으로 분석한다."""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from src.improvement_common import *
from src.group_validation45 import KEEP,load_tools,threshold_grid

def run():
    if (OUT/'diagnostics.json').exists():return
    rows=[];type_rows=[]
    for path in sorted((OUT/'cache').glob('*.parquet')):
        f=pd.read_parquet(path,columns=['Tool','stage','recipe',TARGET]+[f'label_{c}_h600raw' for c in ['f1','f2','f3']]+configs()['A'][2:])
        for state,g in f.groupby(f.stage.eq(0).map({True:'Stage=0',False:'Stage>0'})):
            rows.append({'Tool':path.stem,'state':state,'rows':len(g),'positive':int((g[TARGET]==1).sum()),'negative':int((g[TARGET]==0).sum()),'unknown':int(g[TARGET].isna().sum()),
                         **{c+'_median':float(g[c].median()) for c in configs()['A'][2:]},'recipe_nunique':int(g.recipe.nunique())})
        for code in ['f1','f2','f3']:
            positive=f[f[f'label_{code}_h600raw']==1];negative=f[f[f'label_{code}_h600raw']==0]
            for sensor in configs()['A'][2:]:
                scale=float(negative[sensor].quantile(.75)-negative[sensor].quantile(.25)) if len(negative) else np.nan
                delta=float(positive[sensor].median()-negative[sensor].median()) if len(positive) and len(negative) else np.nan
                type_rows.append({'Tool':path.stem,'fault_code':code,'sensor':sensor,'positive_rows':len(positive),'negative_rows':len(negative),'positive_median':positive[sensor].median(),'negative_median':negative[sensor].median(),'median_difference_over_negative_iqr':delta/scale if scale>0 else None})
    pd.DataFrame(rows).to_parquet(OUT/'tool_state_distribution.parquet',index=False)
    pd.DataFrame(type_rows).to_parquet(OUT/'fault_sensor_contrasts.parquet',index=False)
    old=ROOT/'outputs/stage4/experiments/split_tool_h600/XGB_A'
    result=read_json(old/'result.json');pred=pd.read_parquet(old/'evaluation_predictions.parquet');threshold=result['threshold']
    fault_summary=[]
    for c in ['f1','f2','f3']:
        p=pred[pred[f'label_{c}_h600raw']==1]
        fault_summary.append({'fault_code':c,'rows':len(p),'events':p['next_event_'+c].nunique(),'score_q50':p.score.median(),'score_q95':p.score.quantile(.95),'score_max':p.score.max(),'threshold':threshold,'alarm_rows':int((p.score>=threshold).sum())})
    # 이전 평가 점수는 원인 설명에만 사용한다. 새 임계값은 기존 검증 장비에서만 탐색한다.
    validation=load_tools(['07M01','07M02','08M01','08M02'])
    model=XGBClassifier();model.load_model(old/'model.ubj');model.set_params(device='cpu',n_jobs=4)
    p=validation[KEEP].copy();p['score']=model.predict_proba(matrix(validation,configs()['A']))[:,1]
    _,_,grid=threshold_grid(p)
    save_json(OUT/'old_validation_threshold_grid.json',grid)
    save_json(OUT/'diagnostics.json',{'old_tool_xgb_fault_scores':fault_summary,'old_threshold':threshold,
        'scope':'기존 평가 결과는 실패 원인 설명용이며 새 모델·임계값 선택에 사용하지 않았다. 검증 임계값 민감도 결과를 기존 평가에 적용하지 않았다.',
        'sensor_contrast_limit':'유형별 중앙값 차이는 단변량 기술 통계이며 공정·장비 교란을 제거한 인과효과나 모델 중요도가 아니다.'})
    progress('장비·Stage 분포와 기존 압력 저하 고장 미탐지 진단 완료')

if __name__=='__main__':run()
