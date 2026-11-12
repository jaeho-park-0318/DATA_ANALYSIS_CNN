"""같은 공통 train/validation으로 A/B/C 6개 모델을 새로 학습한다."""
from train_stage2 import main

if __name__=='__main__':main(stage='stage3',families='ABC',cohort='common')
