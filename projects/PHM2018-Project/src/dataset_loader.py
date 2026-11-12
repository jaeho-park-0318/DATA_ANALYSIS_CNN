"""허용된 특징과 하나의 알려진 타깃만 배치로 반환하는 학습용 로더."""
import pyarrow.dataset as ds

from src.dataset_common import DEST, feature_columns, label_columns


def iter_model_batches(target, split, scheme="split_tool", active_only=False, batch_size=65536):
    """미래 메타데이터를 X에서 제외하며 NA·purged 행을 학습에 전달하지 않는다."""
    if target not in label_columns():raise ValueError("기본 정책의 타깃만 선택할 수 있습니다.")
    if split not in ("train","validation","evaluation"):raise ValueError("분할 이름을 확인하세요.")
    if scheme not in ("split_tool","split_time"):raise ValueError("장비/시간 분할 중 하나를 선택하세요.")
    files=[str(p) for p in sorted((DEST/"training").rglob("*.parquet"))]
    if not files:raise FileNotFoundError("검증된 training Parquet가 없습니다.")
    dataset=ds.dataset(files,format="parquet")
    condition=ds.field(target).is_valid() & (ds.field(scheme)==split)
    if active_only:condition=condition & (ds.field("stage")>0)
    names=feature_columns()
    for batch in dataset.to_batches(columns=names+[target],filter=condition,batch_size=batch_size):
        if batch.num_rows==0:continue
        frame=batch.to_pandas()
        yield frame[names],frame[target]
