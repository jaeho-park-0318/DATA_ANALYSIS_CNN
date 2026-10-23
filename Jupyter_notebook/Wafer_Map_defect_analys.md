
# WM811K 기반 CNN Wafer Defect Classification

## 1. 프로젝트 개요

### 1.1 프로젝트 목적

반도체 제조 공정에서는 다양한 원인으로 인해 Wafer에 불량 Die가 발생한다.

이러한 불량 Die의 분포는 특정한 패턴을 형성할 수 있으며, 패턴에 따라 공정 이상이나 결함의 특성을 분석하는 데 활용할 수 있다.

본 프로젝트에서는 **WM811K Wafer Map 데이터셋**을 이용하여 CNN(Convolutional Neural Network) 기반 결함 분류 모델을 구현하였다.

CNN 모델을 통해 8가지 Wafer Defect Pattern을 학습하고, 정답 라벨이 없는 Unknown Wafer의 결함 유형을 예측하는 것을 목표로 하였다.

### 1.2 프로젝트 목표

1. WM811K 데이터셋의 구조와 결함 유형 분석
2. CNN 전용 Python 가상환경 및 Jupyter 커널 구성
3. Wafer Map 이미지 전처리
4. CNN 기반 8종 결함 분류 모델 학습
5. 모델별 Accuracy, Precision, Recall, F1-score 비교
6. CNN 구조 및 Class Weight 변경에 따른 성능 분석
7. Unknown Wafer의 결함 유형 예측
8. Wafer Map 이미지와 예측 결과 시각화

---

## 2. 개발 환경 구축

### 2.1 개발 환경

| 항목 | 사용 환경 |
|---|---|
| 운영체제 | Windows |
| Python | 3.11 |
| 가상환경 | Anaconda / Conda |
| 개발 도구 | Jupyter Notebook |
| 딥러닝 프레임워크 | TensorFlow / Keras |
| 데이터 처리 | NumPy, Pandas |
| 머신러닝 평가 | Scikit-learn |
| 시각화 | Matplotlib |

### 2.2 CNN 전용 가상환경 생성

기존 Anaconda Base 환경에서 발생할 수 있는 라이브러리 충돌을 방지하기 위해 별도의 CNN 전용 환경을 생성하였다.

Anaconda Prompt에서 다음 명령어를 실행하였다.

```bash
conda create -n wafer_cnn python=3.11 -y
conda activate wafer_cnn
```

### 2.3 라이브러리 설치

```bash
python -m pip install --upgrade pip

python -m pip install tensorflow numpy pandas matplotlib scikit-learn ipykernel notebook
```

### 2.4 Jupyter 커널 등록

```bash
python -m ipykernel install --user --name wafer_cnn --display-name "Python (Wafer CNN)"
```

Jupyter Notebook에서 `Python (Wafer CNN)` 커널을 선택하여 실습을 진행하였다.

### 2.5 환경 확인

```python
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("TensorFlow:", tf.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
```

---

## 3. WM811K 데이터셋 분석

### 3.1 데이터셋 개요

WM811K는 반도체 Wafer Map의 불량 Die 분포를 포함하는 데이터셋이다.

각 Wafer는 2차원 배열 형태로 표현되며, Die 상태에 따라 다음 값을 가진다.

| 값 | 의미 |
|---|---|
| 0 | Wafer 외부 영역 |
| 1 | 정상 Die |
| 2 | 불량 Die |

### 3.2 데이터 불러오기

```python
df = pd.read_pickle("WM811K.pkl")

print(df.shape)
print(df.columns)
```

실습에서 확인한 전체 데이터 수는 다음과 같다.

```text
전체 Wafer: 811,457개
```

### 3.3 데이터 라벨 구성

| 구분 | 데이터 수 |
|---|---:|
| Unknown | 638,507 |
| none | 147,431 |
| 8가지 결함 유형 | 25,519 |
| 전체 | 811,457 |

- **Unknown:** 결함 유형의 정답 라벨이 없는 Wafer
- **none:** 정상 패턴으로 라벨링된 Wafer
- **8가지 결함 유형:** CNN 지도학습에 사용한 Wafer

### 3.4 결함 유형별 데이터 수

| 결함 유형 | 데이터 수 |
|---|---:|
| Center | 4,294 |
| Donut | 555 |
| Edge-Loc | 5,189 |
| Edge-Ring | 9,680 |
| Loc | 3,593 |
| Near-full | 149 |
| Random | 866 |
| Scratch | 1,193 |
| **합계** | **25,519** |

데이터를 분석한 결과, 결함 유형별 샘플 수가 크게 차이 나는 **Class Imbalance 문제**를 확인하였다.

특히 Edge-Ring은 9,680개인 반면 Near-full은 149개에 불과하였다.

이러한 데이터 불균형은 CNN 모델이 데이터 수가 많은 클래스에 편향되는 원인이 될 수 있다.

---

## 4. 데이터 전처리

### 4.1 학습 대상 정의

CNN은 다음 8가지 결함 유형을 분류하도록 설계하였다.

```python
class_names = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch"
]
```

`none`과 `Unknown`은 8분류 CNN의 학습 대상에서 제외하였다.

### 4.2 Wafer Map 크기 통일

WM811K의 Wafer Map은 서로 다른 크기를 가지므로 CNN 입력을 위해 크기를 통일하였다.

- 입력 이미지 크기: 64 × 64
- 채널 수: 1
- 이미지 값: 0, 1, 2
- 정규화 범위: 0~1

Wafer Map의 Die 상태가 범주형 값이므로 Nearest Neighbor 방식으로 크기를 조정하였다.

```python
def preprocess_wafer(wafer):
    wafer = np.asarray(wafer, dtype=np.float32)

    wafer = wafer[..., np.newaxis]

    wafer = tf.image.resize(
        wafer,
        (64, 64),
        method="nearest"
    )

    wafer = wafer / 2.0

    return wafer.numpy()
```

### 4.3 Train / Validation / Test 분리

학습 데이터를 다음 세 부분으로 분리하였다.

| 데이터 | 목적 |
|---|---|
| Train | CNN 가중치 학습 |
| Validation | 학습 과정의 일반화 성능 확인 |
| Test | 학습 완료 후 성능 평가 |

동일 Lot에서 나온 Wafer가 서로 다른 데이터 집합에 중복되지 않도록 Lot 기준 분할 방식을 사용하였다.

이는 동일 생산 Lot의 유사한 Wafer 패턴이 Train과 Test에 동시에 포함되어 성능이 과대평가되는 문제를 줄이기 위한 것이다.

---

## 5. CNN V1 모델 구현

### 5.1 CNN 구조

최초 CNN은 다음 구조로 설계하였다.

```text
Input (64×64×1)
       |
       v
Conv2D (32)
       |
MaxPooling2D
       |
Conv2D (64)
       |
MaxPooling2D
       |
Conv2D (128)
       |
MaxPooling2D
       |
GlobalAveragePooling2D
       |
Dense (128, ReLU)
       |
Dropout (0.4)
       |
Dense (8, Softmax)
```

CNN은 Wafer Map의 공간적 결함 분포를 학습하고, Softmax 출력층을 통해 8가지 결함 유형에 대한 예측값을 계산한다.

### 5.2 학습 설정

| 항목 | 설정 |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Sparse Categorical Crossentropy |
| Batch Size | 64 |
| Epoch 상한 | 100 |
| Class Weight | Balanced |
| EarlyStopping | 사용 |
| ReduceLROnPlateau | 사용 |

클래스별 데이터 불균형을 완화하기 위해 Balanced Class Weight를 적용하였다.

### 5.3 초기 학습 결과

초기 CNN 모델에서는 다음과 같은 성능을 확인하였다.

```text
Test Accuracy: 86.42%
```

이후 Epoch 상한을 늘리고 EarlyStopping을 적용한 학습에서 약 89%의 정확도를 얻었다.

해당 모델은 63 Epoch에서 학습이 종료되었다.

EarlyStopping은 Validation Loss가 일정 기간 개선되지 않을 경우 학습을 종료하고, 설정에 따라 최적 가중치를 복원하는 방식이다.

### 5.4 CNN V1 최종 성능

| 결함 유형 | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Center | 0.88 | 0.88 | 0.88 |
| Donut | 0.80 | 0.88 | 0.84 |
| Edge-Loc | 0.88 | 0.91 | 0.90 |
| Edge-Ring | 0.99 | 0.97 | 0.98 |
| Loc | 0.70 | 0.72 | 0.71 |
| Near-full | 0.76 | 1.00 | 0.87 |
| Random | 0.87 | 0.90 | 0.89 |
| Scratch | 0.88 | 0.74 | 0.80 |

```text
Test Accuracy: 약 89%
Macro F1-score: 0.86
Weighted F1-score: 0.89
```

### 5.5 V1 결과 분석

Edge-Ring은 F1-score 0.98로 높은 분류 성능을 보였다.

반면 Loc은 F1-score 0.71로 상대적으로 낮은 성능을 보였다.

Scratch는 초기 모델 대비 Precision이 크게 개선되었지만 Recall은 다소 감소하였다.

이를 통해 전체 Accuracy뿐 아니라 클래스별 Precision, Recall, F1-score를 함께 분석할 필요가 있음을 확인하였다.

---

## 6. CNN V2 모델 구현

### 6.1 V2 모델의 목적

V1의 성능을 개선하기 위해 다음 항목을 변경하였다.

1. BatchNormalization 추가
2. Learning Rate 감소
3. Class Weight 완화

### 6.2 V2 모델 설정

| 항목 | V1 | V2 |
|---|---|---|
| CNN 구조 | 기본 CNN | BatchNormalization 추가 |
| Learning Rate | 0.001 | 0.0005 |
| Class Weight | Balanced | Square Root Balanced |
| Epoch 상한 | 100 | 100 |

### 6.3 BatchNormalization

BatchNormalization은 CNN 중간 계층의 활성값을 정규화하여 학습을 안정화하는 데 활용된다.

V2에서는 Conv2D 이후 BatchNormalization과 ReLU를 적용하였다.

### 6.4 V2 학습 결과

V2는 26 Epoch에서 EarlyStopping에 의해 학습이 종료되었다.

```text
Test Accuracy: 약 83%
Macro F1-score: 0.77
```

### 6.5 V2 그래프 분석

V2 학습 그래프에서는 다음과 같은 현상을 확인하였다.

- Train Accuracy는 약 93%까지 상승
- Validation Accuracy는 큰 폭으로 변동
- Train Loss는 지속적으로 감소
- Validation Loss는 중반 이후 상승하는 경향

이는 학습 데이터에 대한 성능은 높아졌지만 Validation 데이터에 대한 일반화 성능은 개선되지 않는 **Overfitting 현상**으로 해석하였다.

### 6.6 V2 결과 분석

V2는 V1보다 전체 Accuracy와 Macro F1-score가 모두 감소하였다.

특히 Scratch의 F1-score는 0.80에서 0.50으로 크게 감소하였다.

다만 BatchNormalization, Learning Rate, Class Weight를 동시에 변경했으므로 성능 하락의 원인을 특정 요소 하나로 확정할 수는 없다.

---

## 7. CNN V3 모델 구현

### 7.1 V3 모델의 목적

V2의 성능 하락 원인을 분석하기 위해 V1의 CNN 구조로 돌아가 Class Weight만 완화하는 실험을 진행하였다.

V3의 주요 목적은 다음과 같다.

- V1의 CNN 구조 유지
- BatchNormalization 제거
- Square Root Balanced Class Weight 적용
- 전체 정확도 및 소수 클래스 성능 개선

### 7.2 V3 모델 설정

| 항목 | 설정 |
|---|---|
| CNN 구조 | V1과 동일 |
| BatchNormalization | 미사용 |
| Learning Rate | 0.001 |
| Class Weight | Square Root Balanced |
| Epoch 상한 | 100 |
| Batch Size | 64 |
| EarlyStopping | patience=12 |

### 7.3 Class Weight 완화

Balanced Class Weight는 데이터가 적은 클래스에 높은 가중치를 부여한다.

V3에서는 이 가중치에 제곱근을 적용하여 클래스 간 가중치 차이를 완화하였다.

```python
soft_class_weights = {
    k: float(np.sqrt(v))
    for k, v in balanced_weights.items()
}
```

이를 통해 특정 소수 클래스를 지나치게 예측하는 현상을 완화할 수 있는지 실험하였다.

### 7.4 V3 학습 결과

V3는 최대 Epoch인 100까지 학습하였다.

```text
총 학습 Epoch: 100
최적 Epoch: 100
최고 Validation Accuracy: 89.09%
최저 Validation Loss: 0.2863
Test Accuracy: 약 90%
Macro F1-score: 0.87
Weighted F1-score: 0.90
```

### 7.5 V3 클래스별 성능

| 결함 유형 | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Center | 0.92 | 0.87 | 0.89 |
| Donut | 0.88 | 0.87 | 0.87 |
| Edge-Loc | 0.89 | 0.91 | 0.90 |
| Edge-Ring | 0.98 | 0.98 | 0.98 |
| Loc | 0.75 | 0.75 | 0.75 |
| Near-full | 0.81 | 1.00 | 0.90 |
| Random | 0.91 | 0.90 | 0.91 |
| Scratch | 0.67 | 0.80 | 0.73 |

### 7.6 V3 학습 그래프 분석

V3의 Accuracy 및 Loss 그래프에서는 다음 특징을 확인하였다.

**Accuracy**

- Train Accuracy는 약 92%까지 상승
- Validation Accuracy는 약 89% 수준까지 상승
- 두 정확도 사이의 차이는 약 3%p
- V2보다 Validation Accuracy가 안정적으로 유지됨

**Loss**

- Train Loss는 지속적으로 감소
- Validation Loss도 전반적으로 감소
- Epoch 100에서 최저 Validation Loss 기록
- V2와 같은 뚜렷한 Validation Loss 상승 현상은 관찰되지 않음

따라서 V3는 V2보다 안정적인 일반화 성능을 보였다.

다만 Train과 Validation 사이의 차이가 존재하므로 약한 과적합 가능성은 남아 있다.

---

## 8. CNN V1 / V2 / V3 성능 비교

### 8.1 모델별 전체 성능

| 평가 항목 | CNN V1 | CNN V2 | CNN V3 |
|---|---:|---:|---:|
| Test Accuracy | 약 89% | 약 83% | 약 90% |
| Macro F1-score | 0.86 | 0.77 | 0.87 |
| Loc F1-score | 0.71 | 0.66 | 0.75 |
| Scratch F1-score | 0.80 | 0.50 | 0.73 |
| 학습 Epoch | 63 | 26 | 100 |

### 8.2 비교 결과

**CNN V1**

- 기본 CNN 구조
- 비교적 안정적인 학습
- Scratch 분류 성능이 우수
- Loc 분류 성능 개선 필요

**CNN V2**

- BatchNormalization 추가
- Learning Rate 및 Class Weight 변경
- Validation Loss 불안정
- 전체 성능 하락

**CNN V3**

- V1 구조 유지
- Class Weight 완화
- 전체 Accuracy 약 90% 달성
- Macro F1-score 0.87 달성
- Loc F1-score 0.75로 개선
- Scratch F1-score는 V1보다 감소

### 8.3 모델 개선 과정에서 확인한 점

1. Epoch 증가가 항상 정확도 향상으로 이어지는 것은 아니다.
2. Validation Loss를 기준으로 과적합 여부를 판단해야 한다.
3. BatchNormalization을 추가해도 성능이 반드시 개선되지는 않는다.
4. Class Weight 변경은 결함 유형별 Precision과 Recall에 영향을 줄 수 있다.
5. 전체 Accuracy뿐 아니라 Macro F1-score와 클래스별 F1-score가 중요하다.
6. 성능 개선 원인을 정확하게 분석하려면 한 번에 하나의 설정을 변경하는 실험이 필요하다.

---

## 9. Unknown Wafer 분류

### 9.1 Unknown 데이터 확인

초기 실습에서는 Unknown 데이터를 문자열 `"Unknown"`으로 검색하였으나 결과가 0개로 출력되는 문제가 발생하였다.

데이터를 확인한 결과, 현재 전처리된 DataFrame에서는 Unknown 라벨이 `0`으로 저장되어 있었다.

```text
failure_label

0             638507
none          147431
Edge-Ring       9680
Edge-Loc        5189
Center          4294
Loc             3593
Scratch         1193
Random           866
Donut            555
Near-full        149
```

따라서 Unknown 데이터 추출 코드를 수정하였다.

```python
df_unknown = df[
    df["failure_label"].astype(str) == "0"
].copy()
```

### 9.2 Unknown 분류 방식

학습된 CNN V3를 이용하여 Unknown Wafer Map을 입력하고, Softmax 출력값이 가장 높은 결함 유형을 예측하였다.

```text
Unknown Wafer Map
        |
        v
64×64 Resize / Normalization
        |
        v
CNN V3
        |
        v
Softmax (8 Classes)
        |
        v
Predicted Defect Type
        |
        v
Confidence 출력
```

### 9.3 Unknown 예측 코드

```python
unknown_sample = df_unknown.head(100).copy()

X_unknown = np.stack([
    preprocess_wafer(w)
    for w in unknown_sample["waferMap"]
])

pred_prob = model_v3.predict(
    X_unknown,
    batch_size=64
)

pred_class = np.argmax(pred_prob, axis=1)
pred_conf = np.max(pred_prob, axis=1)

unknown_sample["predicted_type"] = [
    class_names[i] for i in pred_class
]

unknown_sample["confidence"] = pred_conf
```

### 9.4 예측 결과 시각화

Wafer Map 이미지와 함께 예측된 결함 유형 및 Confidence를 표시하였다.

```python
fig, axes = plt.subplots(4, 5, figsize=(15, 12))

for i, ax in enumerate(axes.flat):

    wafer = unknown_sample["waferMap"].iloc[i]
    pred = unknown_sample["predicted_type"].iloc[i]
    conf = unknown_sample["confidence"].iloc[i]

    ax.imshow(
        wafer,
        cmap="viridis",
        vmin=0,
        vmax=2
    )

    ax.set_title(
        f"{pred}\nConfidence: {conf:.1%}"
    )

    ax.axis("off")

plt.tight_layout()
plt.show()
```

### 9.5 전체 Unknown 데이터 분류

638,507개의 Unknown Wafer를 한 번에 메모리에 올리면 메모리 부족 문제가 발생할 수 있다.

따라서 Batch 단위로 나누어 예측하는 방법을 사용하도록 코드를 구성하였다.

전체 분류 결과는 CSV 파일로 저장할 수 있도록 하였다.

**주의:** 전체 Unknown 데이터에 대한 실제 예측 정확도는 정답 라벨이 없으므로 계산할 수 없다.

---

## 10. PPT 형식의 예측 결과 시각화

제공된 프로젝트 예시 PPT의 마지막 슬라이드를 참고하여 Wafer Map 이미지와 CNN 예측 결과를 함께 표시하는 코드를 구성하였다.

### 10.1 Test 데이터 결과

Test 데이터는 실제 정답 라벨이 존재하므로 다음 항목을 비교할 수 있다.

- 실제 결함 유형
- CNN 예측 결함 유형
- Confidence
- Correct / Prediction Error

이를 통해 CNN이 올바르게 분류한 Wafer와 오분류한 Wafer를 시각적으로 비교할 수 있다.

### 10.2 Unknown 데이터 결과

Unknown 데이터는 실제 정답이 없으므로 다음 항목을 표시한다.

- Wafer Map 이미지
- CNN 예측 결함 유형
- Confidence

Unknown Wafer의 경우 Correct 또는 Prediction Error 여부를 확정할 수 없다.

---

## 11. 모델 저장 및 활용

### 11.1 모델 저장

```python
model_v1.save("wafer_cnn_v1_89.keras")

model_v3.save("wafer_cnn_v3_final.keras")
```

실습에서는 V1과 V3를 비교 대상으로 보존하였다.

### 11.2 저장된 모델 불러오기

```python
import tensorflow as tf

model = tf.keras.models.load_model(
    "wafer_cnn_v3_final.keras"
)
```

### 11.3 모델 활용

학습된 모델을 사용하면 새로운 Wafer Map을 입력받아 다음 정보를 출력할 수 있다.

1. 예측 결함 유형
2. 결함 유형별 Softmax 출력값
3. 최고 예측 Confidence
4. Wafer Map 시각화 결과

---

## 12. 프로젝트의 한계

### 12.1 데이터 불균형

Near-full, Donut, Scratch 등 일부 결함 유형은 데이터 수가 적다.

특히 Near-full은 Test 데이터가 13개에 불과하므로 Recall이 1.00이더라도 일반화 성능이 충분하다고 단정할 수 없다.

### 12.2 Unknown의 실제 정답 부재

Unknown 데이터는 실제 결함 유형이 제공되지 않으므로 CNN 예측 결과가 정확한지 직접 검증할 수 없다.

### 12.3 정상 Wafer 분류 불가

현재 모델은 8가지 결함 유형만 학습하였다.

따라서 정상 Wafer가 입력되더라도 8가지 결함 중 하나로 분류할 수 있다.

이를 해결하기 위해서는 `none`을 포함한 9분류 모델을 추가로 구현할 필요가 있다.

### 12.4 Softmax Confidence의 한계

Softmax 출력값이 높다고 해서 실제 예측이 반드시 정확하다는 의미는 아니다.

특히 학습하지 않은 새로운 유형의 결함에 대해서도 높은 Confidence가 출력될 수 있다.

### 12.5 Test 데이터 반복 활용

여러 모델의 Test 성능을 확인하며 모델을 개선했기 때문에 최종 성능 평가에는 주의가 필요하다.

추가 연구에서는 Validation 데이터를 기준으로 모델을 선택하고, 별도의 미사용 Test 데이터에서 최종 평가하는 방식이 바람직하다.

---

## 13. 향후 개선 방향

### 13.1 추가 Epoch 학습

V3는 Epoch 100에서 최저 Validation Loss를 기록하였다.

따라서 학습을 추가로 진행하여 Validation Loss가 더 개선되는지 확인할 수 있다.

단, 추가 학습이 Test Accuracy 향상을 보장하지는 않는다.

### 13.2 Data Augmentation

Wafer Map의 결함 형태를 보존하는 범위에서 데이터 증강을 적용할 수 있다.

예시:

- 90도 회전
- 좌우 반전
- 상하 반전

다만 결함의 방향이나 위치가 중요한 경우 데이터 증강이 오히려 분류에 불리할 수 있으므로 별도 검증이 필요하다.

### 13.3 OverSampling

소수 클래스의 학습 샘플을 늘려 Class Imbalance 문제를 완화할 수 있다.

단, OverSampling은 Train 데이터에만 적용해야 하며 Validation 및 Test 데이터에는 적용하지 않는다.

### 13.4 9분류 CNN

기존 8가지 결함 유형에 정상 Wafer인 `none`을 추가한다.

이를 통해 정상 Wafer와 결함 Wafer를 함께 분류하는 모델로 확장할 수 있다.

### 13.5 실제 공정 결함 이미지 분석

향후에는 SEM 또는 Optical Defect 이미지 데이터를 활용하여 다음과 같은 결함을 분류하는 CNN 모델로 확장할 수 있다.

- Particle
- Residue
- Bridge
- Pattern Collapse
- Scratch
- Void

이후 공정 정보와 EDS 등의 분석 데이터를 결합하면 결함 원인 후보를 분석하는 시스템으로 발전시킬 수 있다.

다만 Wafer Map 패턴과 실제 SEM 결함 이미지가 동일한 Wafer 또는 공정 이력으로 연결되지 않는다면 직접적인 원인 분석에는 한계가 있다.

---

## 14. 최종 결론

본 프로젝트에서는 WM811K Wafer Map 데이터셋을 이용하여 CNN 기반 8종 결함 분류 모델을 구현하고, 모델 구조 및 학습 설정 변화에 따른 성능을 비교하였다.

초기 CNN 모델에서는 약 86.42%의 Test Accuracy를 확인하였으며, 학습 조건을 조정한 V1 모델에서 약 89%의 정확도를 달성하였다.

이후 BatchNormalization을 추가한 V2 모델에서는 과적합 현상이 발생하면서 정확도가 약 83%로 감소하였다.

반면 V1 구조를 유지하고 Class Weight를 완화한 V3 모델에서는 약 90%의 Test Accuracy와 0.87의 Macro F1-score를 달성하였다.

특히 V3에서는 Loc의 F1-score가 V1의 0.71에서 0.75로 개선되었으며, 전체적인 분류 성능도 소폭 향상되었다.

이러한 결과를 통해 CNN의 성능은 단순히 Epoch 수를 늘리거나 모델 구조를 복잡하게 만드는 것만으로 향상되지 않으며, 데이터 불균형 처리와 학습 설정이 중요한 영향을 미친다는 점을 확인하였다.

최종적으로 학습된 CNN V3 모델을 활용하여 정답 라벨이 없는 Unknown Wafer의 결함 유형을 예측하는 코드를 구현하였으며, Wafer Map 이미지와 예측 결과를 함께 시각화할 수 있도록 구성하였다.

본 실습은 반도체 Wafer Defect Pattern을 딥러닝으로 분류하는 기초 모델을 구현하고, 향후 실제 공정 결함 이미지 및 공정 이력 기반 분석으로 확장하기 위한 기반을 마련했다는 의의가 있다.

---

## 15. 참고 자료

- WM811K Wafer Map Dataset
- 제공된 `Project_Example.pptx`
- TensorFlow / Keras Documentation
- Scikit-learn Documentation

## 프로젝트 핵심 키워드

`Semiconductor` `Wafer Map` `WM811K` `CNN` `TensorFlow` `Keras` `Defect Classification` `Deep Learning` `Class Imbalance` `Transfer Learning`
