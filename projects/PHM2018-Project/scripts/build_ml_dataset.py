"""대표 장비 검증 후 전체 장비로 확장하는 로컬 3단계 실행 진입점."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8")

from src.build_labels import build as labels
from src.build_features import build as features
from src.dataset_common import DATA, DEST, REPORT, POLICY, feature_columns, save_json, snapshot, ensure_compatible


def main():
    if Path(sys.prefix).name.lower()!="phm2018":raise RuntimeError("phm2018 환경에서 실행하세요.")
    ensure_compatible()
    before=snapshot()
    tools=[p.stem[:6].replace("_","") for p in sorted((DATA/"train").glob("*.csv"))]
    label_summaries=[];feature_summaries=[]
    for tool in tools:
        label_summaries.extend(labels([tool]));feature_summaries.extend(features([tool]))
    assert before==snapshot()
    save_json(REPORT/"label_build.json",label_summaries)
    save_json(REPORT/"feature_build.json",feature_summaries)
    save_json(DEST/"manifest.json",{"policy":POLICY,"tools":tools,"source_snapshot":before,
                                   "label_summaries":label_summaries,"feature_summaries":feature_summaries,
                                   "feature_columns":feature_columns(),"python":sys.executable})
    print("전체 장비 라벨·특징 구축 완료",flush=True)


if __name__=="__main__":main()
