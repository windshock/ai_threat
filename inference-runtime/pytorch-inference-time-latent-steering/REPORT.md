# PyTorch Inference-time Latent Steering을 통한 LLM Safety 우회 사례 분석 (개정판)

## 1. 개요 (Executive Summary)

본 보고서는 **PyTorch 표준 API를 활용한 inference-time latent steering(activation patching / representation engineering)** 기법을 통해,
**대규모 언어 모델(LLM)의 safety refusal이 우회될 수 있음을 보여주는 공개 데모 사례**를 분석한다.

해당 사례는 새로운 이론적 기여를 제시한 연구라기보다는,
**이미 학계에서 논의되어 온 개념이 “현역 상용급 모델 + 재현 가능한 코드” 수준에서 현실화되었음을 명확히 보여준 실증 데모**라는 점에서 의미를 갖는다.

특히, **LLM 안전의 Single Point of Failure(SPOF)가 모델 자체가 아니라 inference runtime에 있음**을 구체적으로 드러낸다는 점에서, 보안·운영 관점의 시사점이 크다.

---

## 2. 기술적 배경 및 개념 정리

### 2.1 Latent Steering의 정의

Latent steering은 다음과 같은 특성을 갖는 **추론 시점(inference-time) 개입 기법**이다.

- 모델의 forward pass 중
- 특정 레이어의 hidden state(activation)를
- 외부에서 읽거나 수정하여
- 이후 추론 경로를 변화시킴

이는 다음과 명확히 구분된다.

| 구분 | Latent Steering |
| --- | --- |
| 학습 단계 개입 | ❌ |
| 가중치 변경 | ❌ |
| 지속성 | 휘발성 |
| 재시작 후 영향 | 없음 |

➡️ **Fine-tuning이나 RLHF가 아닌 runtime intervention**

---

### 2.2 기존 연구와의 관계

관련 선행 연구 및 공개 작업에는 다음이 포함된다.

- *Activation patching / causal tracing* (Anthropic 계열 연구)
- *Representation engineering*
- *Inference-time category-wise safety steering* (arXiv)
- Sparse Autoencoder(SAE)를 활용한 concept vector 주입
  - 예: Hugging Face 연구자의 “Eiffel Tower” concept injection 데모 (PyTorch 기반)

이러한 연구들은 **latent space 개입의 가능성 자체는 이미 입증**해 왔다.

---

## 3. PyTorch 기반 구현의 의미

### 3.1 구현 방식의 핵심

이번 사례의 기술적 핵심은 다음과 같다.

- PyTorch `register_forward_hook` 사용
- 특정 Transformer 레이어 출력 tensor 가로채기
- 외부 latent vector / memory를 residual stream에 주입
- 이후 레이어로 수정된 activation 전달

📌 **모든 과정은 PyTorch의 공개·표준 API 범위 내에서 수행**

---

### 3.2 “왜 이번 사례가 주목되는가”

중요한 점은 **기술 자체가 아니라 조합**이다.

- 추상적 논문 ❌
- toy model ❌
- 내부 코드 ❌

대신:

- ✅ **현역 상용급 모델 (Gemma-3 12B)**
- ✅ **명확한 safety refusal 우회 효과**
- ✅ **스크린샷 + 코드 기반 재현 가능성**
- ✅ **추론 런타임에서의 직접 개입**

➡️ 이 조합이 **공개적으로 명확히 제시된 사례는 상대적으로 드물었다**.

---

### 3.3 (직관 설명) 훅은 원래 뭘 하려고 쓰이고, 왜 “오용”이 가능한가

#### 훅의 원래 용도: “관찰/계측/디버깅”을 위한 표준 도구

PyTorch의 `register_forward_hook` 같은 훅은 본래 **모델의 중간 값을 보기 위해** 제공되는 기능이다. 대표적으로:

- 디버깅: 특정 레이어에서 NaN이 생기는지, 텐서 shape이 맞는지 확인
- 계측/로깅: 레이어별 activation norm/분포를 모니터링
- 특성 추출: 특정 레이어 representation을 뽑아 다운스트림 작업에 사용
- 연구/분석: activation patching, causal tracing 같은 해석 실험

즉, 훅은 “가중치(파라미터)를 바꾸지 않고도” 추론 중간을 들여다볼 수 있게 해 주는 **정상적인 개발 편의 기능**이다.

#### 왜 오용이 가능한가: 훅은 “중간값을 바꾸는 공식 통로”이기 때문

문제는 훅이 **보기(read)**만 하는 것이 아니라, 구현에 따라 **바꾸기(write)**도 가능하다는 점이다.
LLM의 출력은 최종적으로 “다음 토큰 확률(logits)”에서 결정되는데, 이 logits는 내부 hidden state의 함수다.

- hidden state를 조금만 바꾸면 → logits가 달라지고
- logits가 달라지면 → 다음 토큰 선택이 달라지고
- 한 번 토큰이 달라지면 → 이후 입력 자체가 바뀌어 차이가 눈덩이처럼 커진다

따라서 훅을 통해 특정 레이어의 hidden을 조작하면, “문체/톤”처럼 무해한 특성뿐 아니라,
모델의 **거부(refusal) 행동** 같은 정책적 행동도 (의도에 따라) 영향을 받을 수 있다.

#### 보안 관점 요약: 모델이 아니라 런타임이 신뢰 경계(TCB)가 된다

훅은 프레임워크가 제공하는 **정상 기능**이지만, 공격 관점에서는 “living-off-the-land” 도구로도 볼 수 있다.
즉, 모델이 아무리 정렬되어 있어도, 모델을 실행하는 런타임이 신뢰할 수 없는 상태(권한 있는 주입/플러그인/커스텀 코드 경로)라면
정책 준수는 깨질 수 있다. 이것이 “SPOF는 모델이 아니라 inference runtime”이라는 진단이 성립하는 핵심 이유다.

## 4. Gemma-3 12B 사례 분석

### 4.1 관찰된 현상

- 정상 실행 시:
  - 안전 정책에 따라 응답 거부
- Latent steering 적용 시:
  - 형식적 안전 문구 유지
  - **실질적으로는 금지된 내용 생성**

이는 Reddit 등에서 언급된 Gemma-3 계열의
“abliteration(거부 행동 제거)” 데모와도 정합적이다.

---

### 4.2 모델 아키텍처 관점

Gemma-3의 구조적 특성(GeGLU, RMSNorm, residual stream)은
**inference runtime에서의 activation 개입을 기술적으로 어렵게 만들지 않는다**는 점이 확인된다.

---

## 5. vLLM 등 Inference Runtime과의 관계

### 5.1 개념적 범위

본 보고서에서 말하는 “runtime”은 다음을 포괄한다.

- transformers + PyTorch eager
- custom PyTorch inference server
- DeepSpeed inference
- Triton server
- vLLM (제한적)

---

### 5.2 vLLM의 위치

- Python-level `register_forward_hook`은 기본적으로 어려움
- 그러나:
  - logits post-processing
  - attention bias
  - KV-cache manipulation
  - scheduler 단계 개입
    등 **다른 조작 지점은 여전히 존재**

➡️ **방어가 아니라 공격 표면이 아래로 이동한 것**

---

## 6. 연구적 위치 평가 (Updated)

### 6.1 새로움에 대한 정확한 평가

- ❌ “최초 개념 제시”
- ❌ “새 이론”
- ✅ **PyTorch 기반으로 재현 가능한 safety bypass 데모가 점점 누적되는 흐름의 일부**
- ✅ 2023년 Andy Zou(@andyzou_jiaming)의 **“Super Suffixes” 작업이 공개적·실증적 선행 사례로서 seminal한 위치**

Zou의 작업은:

- adversarial suffix optimization을 통해
- OSS 모델(Vicuna)뿐 아니라
- 상용 모델(ChatGPT/Claude/Bard)의 safety bypass 가능성을
- 코드·웹사이트와 함께 공개적으로 입증

➡️ 이번 Gemma-3 사례는 **그 흐름이 runtime-level latent intervention까지 확장되었음을 보여주는 후속 증거**로 위치 지을 수 있다.

---

## 7. 보안 관점 해석

### 7.1 핵심 진단

> **LLM 안전의 SPOF는 모델이 아니라 inference runtime이다**

- Alignment는 필요조건일 뿐 충분조건이 아님
- Runtime이 신뢰 경계를 벗어나면:
  - safety head
  - refusal behavior
  - policy alignment
    모두 무력화 가능

---

### 7.2 기존 보안과의 구조적 유사성

| LLM | 전통 보안 |
| --- | --- |
| inference runtime | userland / kernel |
| latent hook | API hooking |
| 정상 API 악용 | living-off-the-land |
| alignment 무력화 | policy enforcement bypass |

➡️ **AI 보안은 새로운 문제가 아니라, 익숙한 보안 문제의 재등장**

---

## 8. 실무적 시사점

- “모델 alignment 강화”만으로는 불충분
- Inference runtime을 **Trusted Computing Base(TCB)** 로 인식해야 함
- Custom inference server는 특히 취약
- vLLM 등 optimized runtime도 **완전한 해결책은 아님**

---

## 9. 결론 요약

> **이 사례는 완전히 새로운 연구는 아니다. 그러나 ‘이제는 추상적 논의가 아니라, 재현 가능한 현실 문제’임을 분명히 했다.**

---

## 10. 한 문장 요약 (Updated)

> **“LLM 안전은 모델의 문제가 아니라, 모델을 실행하는 런타임의 신뢰 문제다.”**


