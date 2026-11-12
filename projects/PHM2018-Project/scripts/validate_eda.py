"""집계 불변식, 실제 이벤트 관측 충분성, 라벨 경계 및 Notebook 실행을 확인한다."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import nbformat
import numpy as np
import pandas as pd
from src.eda_common import DATA, OUT, read_json, load_faults, snapshot
from src.eda_detail import coverage, horizon_labels


def main():
    result=read_json("overview.json");detail=read_json("detail.json")
    tools=pd.DataFrame(result["tools"])
    assert tools[tools.split=="train"].rows.sum()==82_189_440
    assert tools[tools.split=="test"].rows.sum()==7_198_948
    assert tools.time_reversals.sum()==0
    assert len(result["schemas"])==65
    assert all(x["ordered_time_keys_equal"] for x in read_json("time_keys.json")["files"])
    faults=load_faults();unique=faults.drop_duplicates(["Tool","time","fault_raw"])
    assert len(faults)==1236 and len(unique)==1211
    assert unique.fault_name.value_counts().to_dict()=={
        "FlowCool Pressure Dropped Below Limit":741,
        "Flowcool Pressure Too High Check Flowcool Pump":333,"Flowcool leak":137}
    ttf=pd.DataFrame(result["ttf"])
    assert (ttf["rows"]==ttf["missing"]+ttf["finite"]).all()
    assert ttf.mismatch.sum()==3489
    assert sum(x["mismatch_excluding_tag"] for x in read_json("tagged_ttf.json"))==0
    assert ttf.missing_with_future_fault.sum()==0
    assert ttf.no_future_fault_finite.sum()==0
    pre=pd.read_json(OUT/"pre_event.json");ft=detail["selected_event"]["time"]
    assert (pre.time<ft).all()
    for minutes in (5,10,30,60):
        sub=pre[pre.time>=ft-minutes*60]
        assert coverage(sub,ft-minutes*60,ft,detail["mode_dt"])["sufficient"]
    assert not detail["secondary_event"]["windows"][-1]["sufficient"]
    # 실제 결과와 구분된 라벨 함수 경계 단위 테스트.
    times=np.array([0,5,10,20,25],dtype=float)
    conservative=horizon_labels(times,np.array([10]),20,5)
    assert conservative.isna().tolist()==[True,False,True,True,True]
    checked=horizon_labels(times,np.array([10]),20,5,followup_complete=np.ones(5,dtype=bool))
    assert checked[0]==0 and checked[1]==1 and checked[2:].isna().all()
    conflicting=horizon_labels(np.array([5]),np.array([10]),20,5,valid_ttf=np.array([99]))
    assert conflicting.isna().all()
    book_status=[]
    for path in sorted((ROOT/"notebooks").glob("*.ipynb")):
        book=nbformat.read(path,as_version=4);nbformat.validate(book)
        codes=[c for c in book.cells if c.cell_type=="code"]
        assert all(c.execution_count is not None for c in codes)
        assert not any(o.output_type=="error" for c in codes for o in c.outputs)
        images=sum("image/png" in o.get("data",{}) for c in codes for o in c.outputs)
        assert images>0
        book_status.append({"Notebook":path.name,"실행 코드 셀":len(codes),"그래프 출력":images,"오류":0})
    initial=json.loads((ROOT/"reports/local/inspection.json").read_text(encoding="utf-8-sig"))
    current=snapshot()
    assert current=={x["path"]:[x["bytes"],x["mtime_ns"]] for x in initial["inventory"]}
    report={"검증":"통과","Notebook":book_status,"원본_크기_수정시각_유지":True,
            "설명":"전체 원본 해시 검사는 하지 않았으며 크기와 수정 시각을 비교함"}
    (OUT/"validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
