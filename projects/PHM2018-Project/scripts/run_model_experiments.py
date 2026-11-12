"""600 구간의 본 실험을 먼저 검증한 뒤 고정 설정으로 확장한다."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8")
from src.model_common import OUT,SCHEMES,save_json,preflight
from src.train_baseline import train as baseline
from src.train_xgboost import train as xgboost


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--extensions",action="store_true");args=parser.parse_args()
    if Path(sys.prefix).name.lower()!="phm2018":raise RuntimeError("phm2018 환경에서 실행하세요.")
    preflight();results=[]
    for scheme in SCHEMES:
        results+=baseline(scheme,600);results+=xgboost(scheme,600)
    assert all(r["completed"] and r["metrics"]["evaluation"]["positive"]>0 for r in results)
    save_json(OUT/"primary_gate.json",{"horizon":600,"experiments":len(results),"passed":True,
                                       "rule":"실행 오류 없음·평가 양성 존재·고정 분할; 평가 성능으로 설정을 조정하지 않음"})
    if args.extensions:
        for horizon in (300,1800):
            for scheme in SCHEMES:
                results+=baseline(scheme,horizon);results+=xgboost(scheme,horizon)
    save_json(OUT/"run_status.json",{"experiments":len(results),"primary_completed":True,"extensions_completed":args.extensions})


if __name__=="__main__":main()
