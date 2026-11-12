"""장비 그룹 CV와 별도 시간 분할. 분할 이름은 미래 학습 단계의 역할만 정의한다."""
from src.common import *

ROLES = ["train", "early_stopping", "threshold_selection", "evaluation"]


def tool_roles(tool):
    # 같은 번호의 M01/M02는 함께 묶고, 인접 번호 2개를 외부 그룹으로 둔다.
    group = (int(tool[:2]) - 1) // 2
    return {f"fold_{fold}": ("evaluation" if group == fold else
                              "threshold_selection" if group == (fold + 1) % 5 else
                              "early_stopping" if group == (fold + 2) % 5 else "train")
            for fold in range(5)}


def time_roles(t, lo, hi):
    edges = np.r_[lo, lo + (hi - lo) * np.array([.5, .7, .85]), hi + 1]
    roles = np.full(len(t), "purged", dtype=object)
    for i, role in enumerate(ROLES):
        valid = (t - 900 >= edges[i]) & (t + 1800 < edges[i + 1])
        roles[valid] = role
    return roles
