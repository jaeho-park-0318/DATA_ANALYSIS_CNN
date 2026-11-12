"""로컬 phm2018 커널로 5개 Notebook을 생성하고 실제 실행한다."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import os
import time
import nbformat as nb
from nbclient import NotebookClient
from src.common import *

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--only',nargs='*');args=parser.parse_args()
    jupyter=ROOT/'outputs/jupyter';kernel=jupyter/'kernels/phm2018-v3';kernel.mkdir(parents=True,exist_ok=True)
    dump_json(kernel/'kernel.json',{'argv':[CFG['python'],'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'PHM2018 V3 (로컬)','language':'python'})
    os.environ['JUPYTER_PATH']=str(jupyter)
    os.environ['JUPYTER_RUNTIME_DIR']=str(ROOT/'outputs/jupyter/runtime')
    os.environ['IPYTHONDIR']=str(ROOT/'outputs/ipython')
    os.environ['MPLCONFIGDIR']=str(ROOT/'outputs/.mplconfig')
    common="""from pathlib import Path
import sys, json
ROOT = Path.cwd()
if ROOT.name == 'notebooks': ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
import pandas as pd
from IPython.display import display, Image, Markdown
from src.common import TABLES, CFG
from src.detail import get_detail
from src.visuals import detail_figure
assert Path(sys.executable).resolve() == Path(CFG['python']).resolve()
print('실제 실행 Python:', sys.executable)
print('시간: raw / 연구 진단용 / 모델 학습 없음')
"""
    notebooks=[
        ('06_tool_overview','전체 장비 현황',[
            "tools = pd.read_parquet(TABLES/'tool_summary.parquet')\ndisplay(tools)\nprint('원본 합계:', f'{tools.raw_rows.sum():,}', '행')",
            "display(Image(filename=str(ROOT/'outputs/figures/tool_overview/raw_rows.png')))\ndisplay(Image(filename=str(ROOT/'outputs/figures/tool_overview/ttf_null.png')))",
            "display(Markdown('TTF Null은 정상으로 변환하지 않습니다. 전체 집계의 분모는 원본 행이며, 이벤트 수와 구분합니다.'))"]),
        ('07_timeline_explorer','관측 타임라인과 구간 탐색',[
            "tool = '01M01'\nsegments = pd.read_parquet(TABLES/'time_segment_summary.parquet')\ndisplay(segments[segments.Tool.eq(tool) & segments.kind.eq('fixed')].head(20))",
            "display(Image(filename=str(ROOT/'outputs/figures/timeline/all_tools.png')))\ndisplay(Image(filename=str(ROOT/'outputs/figures/timeline/gap_distribution.png')))",
            "# 이 값들을 바꾸어 다른 실제 구간을 조회합니다. 최대20,000 raw입니다.\ncases = json.loads((ROOT/'outputs/case_catalog.json').read_text(encoding='utf-8'))\ncase = next(c for c in cases if c['Tool']==tool and c['category']=='긴 센서 공백')\ndetail = get_detail(tool, case['start'], case['end'])\ndisplay(pd.DataFrame(detail['gaps']))\ndisplay(pd.DataFrame(detail['anchor_table']).head(15))\nprint(detail['load_audit'])"]),
        ('08_fault_event_explorer','고장 이벤트별 원본 센서 탐색',[
            "tool = '03M02'\nevents = pd.read_parquet(TABLES/'fault_event_summary.parquet')\ndisplay(events[events.Tool.eq(tool)].head(20))",
            "# event_id와 lookback_raw를 변경하면 해당 원본 구간을 다시 읽습니다.\nevent = events[events.Tool.eq(tool) & events.coverage_pre300.ge(.9)].iloc[0]\nevent_id = event.event_id\nlookback_raw = 1800\ndetail = get_detail(tool,event.fault_time-lookback_raw,event.fault_time+120,event_id)\nfig = detail_figure(detail)\nfig.suptitle(event_id+' / 실제 원본 해상도100% / 시간 raw')\nfig.text(.02,.01,detail['source']+' / 원본 '+str(detail['raw_rows'])+'행 / 범위 내 고유 이벤트 '+str(detail['unique_events_in_range'])+'개',fontsize=8)\nfig.tight_layout(rect=(0,.05,1,.96))\nimport io\nbuffer = io.BytesIO()\nfig.savefig(buffer,format='png',dpi=120)\ndisplay(Image(data=buffer.getvalue()))\nimport matplotlib.pyplot as plt\nplt.close(fig)",
            "display(pd.DataFrame(detail['exclusions']))\ndisplay(pd.DataFrame(detail['anchor_table']).head(20))\ndisplay(Markdown('고장 이후는 참고 표시입니다. 정지 시각 이전의 변화가 인과관계나 실제 이상 시작을 입증하지는 않습니다.'))"]),
        ('09_label_visualization','라벨 정책과 경계 검증',[
            "labels = pd.read_parquet(TABLES/'label_status_summary.parquet')\nlevel = 'anchor'\ndisplay(labels[labels.Tool.eq('01M01') & labels.H.eq(600) & labels.sample_level.eq(level)])",
            "display(Image(filename=str(ROOT/'outputs/figures/label_analysis/horizon_policies.png')))\ndisplay(Image(filename=str(ROOT/'outputs/figures/label_analysis/ttf_null_relation.png')))",
            "# 실제 이벤트 주변: 최소 선행시간 경계와 통합 우선순위 확인\nevents = pd.read_parquet(TABLES/'events.parquet')\nevent = events[events.Tool.eq('01M01')].iloc[0]\nfrom src.labels import make_labels, STATUS_NAMES\nfrom src.query import manifest\ntimes = [event.fault_time-601,event.fault_time-600,event.fault_time-60,event.fault_time-59,event.fault_time]\nresult,*_ = make_labels(times,events,'01M01',manifest('01M01')['max_time'])\ndisplay(pd.DataFrame({'time':times,**{p:STATUS_NAMES[result[p+'__any__h600']] for p in CFG['policies']}}))"]),
        ('10_feature_availability','A/B/C 가용성과 이벤트 보존',[
            "features = pd.read_parquet(TABLES/'feature_availability_summary.parquet')\nselection = features.policy.eq('conservative') & features.H.eq(600) & features.fault_type.eq('any')\ndisplay(features[selection].groupby(['cohort','feature_set'])[['available_anchors','positive_anchors','positive_events']].sum())",
            "display(Image(filename=str(ROOT/'outputs/figures/feature_availability/cohorts.png')))\ndisplay(Image(filename=str(ROOT/'outputs/figures/event_preservation/exclusion_reasons.png')))",
            "reasons = pd.read_parquet(TABLES/'event_exclusion_reasons.parquet')\ndisplay(reasons[reasons.Tool.eq('03M02') & reasons.policy.eq('conservative') & reasons.H.eq(600) & reasons.feature_set.eq('C')].head(20))\ndisplay(Markdown('공통 코호트와 최대 가용 코호트를 구분합니다. 특징 수가 많다는 사실만으로 성능이 좋다고 판단하지 않습니다.'))"])
    ]
    result_path=ROOT/'outputs/validation/notebook_execution.json'
    results=json.loads(result_path.read_text(encoding='utf-8')) if args.only and result_path.exists() else []
    for name,title,codes in notebooks:
        if args.only and name not in args.only:continue
        book=nb.v4.new_notebook();book.metadata['kernelspec']={'name':'phm2018-v3','display_name':'PHM2018 V3 (로컬)','language':'python'}
        book.cells=[nb.v4.new_markdown_cell('# '+title+'\n\n실제 V3 원본 인덱스와 집계만 사용합니다. 원본·V1/V2는 읽기 전용입니다. 대시보드는 `dashboard/app.py`로 실행합니다.'),nb.v4.new_code_cell(common)]
        for code in codes:book.cells.append(nb.v4.new_code_cell(code))
        path=ROOT/'notebooks'/(name+'.ipynb');nb.write(book,path)
        start=time.monotonic();NotebookClient(book,timeout=240,kernel_name='phm2018-v3',resources={'metadata':{'path':str(ROOT)}}).execute()
        nb.write(book,path);codecells=[c for c in book.cells if c.cell_type=='code'];errors=[o for c in codecells for o in c.outputs if o.output_type=='error']
        results=[x for x in results if x['notebook']!=name]
        results.append({'notebook':name,'code_cells':len(codecells),'executed':sum(c.execution_count is not None for c in codecells),'errors':len(errors),'elapsed_seconds':time.monotonic()-start,'python':CFG['python']})
        dump_json(ROOT/'outputs/validation/notebook_execution.json',results);print(name+' 실행 완료',flush=True)

if __name__=='__main__':main()
