"""Meaningful event-rule and frozen-reference checks; synthetic cases only test logic."""
from evaluate import *

def run():
    threshold=.9671916365623474;checks=[]
    g=np.zeros(941,dtype=int);s=np.zeros(941,dtype=np.float32);valid=np.ones(941,bool)
    g[141:144]=2;g[144:147]=1;s[141:147]=1
    e=event_result(g,s,valid,1,1,threshold)
    assert e['first_detection_sample']==163 and e['first_correct_diagnosis_sample']==166
    assert e['first_confirmed_type']==2 and not e['first_confirmed_type_correct'] and e['additional_diagnosis_delay_minutes']==9
    checks.append('wrong first type followed by later correct diagnosis')
    g[141:147]=[1,2,1,2,1,2]
    e=event_result(g,s,valid,1,1,threshold)
    assert e['detected'] and not e['correctly_diagnosed'] and e['first_confirmed_type'] is None
    checks.append('mixed types detect a fault but cannot form3 identical diagnoses')
    valid[141:]=False
    e=event_result(g,s,valid,1,1,threshold)
    assert e['valid_post_windows']==0 and not e['detected'] and not e['correctly_diagnosed']
    checks.append('unavailable inputs do not produce diagnoses')
    allclasses=pd.read_csv(OUT/'tables/per_class_metrics.csv')
    existing=pd.read_csv(ROOT/'reports/tables/phase2/primary_dense_classification_report.csv')
    new=allclasses[allclasses['mode']=='gated21'].sort_values('class')
    for col in ['precision','recall','f1','support']:np.testing.assert_allclose(new[col],existing[col],rtol=0,atol=1e-12)
    checks.append('all21 existing class precision/recall/F1/support reproduced')
    runs=pd.read_csv(OUT/'tables/run_type_results.csv')
    assert not ((~runs.detected)&runs.correctly_diagnosed).any()
    assert runs.additional_diagnosis_delay_minutes.dropna().ge(0).all()
    checks.append('correct diagnosis requires detection and cannot precede detection')
    protected=json.loads((OUT/'sources/protected_before.json').read_text(encoding='utf-8'))
    changed=[name for name,value in protected.items() if sha(ROOT/name)!=value]
    assert not changed
    checks.append('all protected existing notebooks/model/checkpoint/config files unchanged')
    save('sources/test_results.json',{'passed':True,'checks':checks,'synthetic_inputs_used_only_for_logic_tests':True,'changed_files':changed})
    print('Diagnosis checks PASSED',len(checks),flush=True)

if __name__=='__main__':run()
