# 구현 재확인: `REPORT.md` vs 현재 레포

이 문서는 `REPORT.md`(사례 분석 보고서)의 **주요 주장/구현 포인트**가
현재 레포의 **실제 코드 구현**(`activation_hook_demo.py`, `README.md`)과
어느 정도 정합적인지 재확인한 체크리스트입니다.

> 중요: 이 레포는 **“안전한(benign) 스타일/톤 steering 데모”**를 목표로 하며,
> **safety refusal 우회/유해 출력 유도 데모는 포함하지 않습니다.**

---

## 1) PyTorch 표준 API로 inference-time 개입인가?

- **보고서 주장**: `register_forward_hook`로 특정 레이어 activation을 가로채 수정
- **현재 구현**: ✅ 일치
  - `ActivationSteerer`가 `layer.register_forward_hook(...)` 사용
  - block output이 Tensor/tuple인 경우를 모두 처리

---

## 2) 개입 대상이 “특정 Transformer 블록의 hidden state”인가?

- **보고서 주장**: 특정 Transformer 레이어 출력(hidden state/activation) 수정
- **현재 구현**: ✅ 일치
  - `get_transformer_blocks()`가 모델 아키텍처별 block 목록을 찾아,
    중간 레이어(`layer_idx`)를 선택해 hook 적용
  - hook은 `(batch, seq, hidden)` 형태의 hidden state를 대상으로 함

---

## 3) “외부 latent vector”를 residual stream에 주입하는가?

- **보고서 주장**: 외부 latent vector/memory를 residual stream에 주입
- **현재 구현**: ✅ 일치 (2가지 모드)
  - `MODE=vector_add`: `steering_vector v`를 만들고, hook에서 `hidden + alpha * v`를 더함
  - `MODE=cross_read`: “메모리(mem)를 attention처럼 읽어” hidden에 섞는 방식(잔차 형태로 mixing)
    - 메모리는 기본적으로 “cheerful 프롬프트”에서 지정 레이어 hidden을 추출해 구성

---

## 4) steering vector는 어떻게 구성되는가?

- **보고서 맥락**: concept vector/SAE/카테고리 steering 등 다양한 가능성 언급
- **현재 구현**: ✅ “representation engineering”의 한 방식으로 구현
  - `v = hidden(target_prompt) - hidden(neutral_prompt)` 후 normalize
  - 대상은 **무해한 스타일/톤 차이**(neutral vs cheerful)

---

## 5) “safety refusal 우회”가 실제로 재현되는가?

- **보고서 주장**: Gemma 계열에서 safety refusal을 형식적으로 유지하면서 실질적으로 우회되는 관찰
- **현재 구현**: ❌ 불일치 (의도적으로 미구현)
  - 이 레포는 **안전한 문체/톤 steering 데모**만 포함
  - 금지된/유해 컨텐츠 유도를 목표로 하지 않음
  - 따라서 “우회 효과”를 재현/검증하는 프롬프트/평가/증적(스크린샷)은 포함하지 않음

---

## 6) 모델/스케일: “Gemma-3 12B”와 동일한가?

- **보고서 주장**: “Gemma-3 12B” 사례 언급
- **현재 구현**: ❌ 불일치
  - 현재 테스트/실행은 `gpt2` 및 `google/gemma-2-2b-it` 기준
  - `Gemma-3 12B`는 이 레포에서 기본 대상으로 고정되어 있지 않음

---

## 7) runtime 범위(vLLM 등)까지 커버하는가?

- **보고서 주장**: vLLM 등 다양한 runtime에서 공격 표면이 이동
- **현재 구현**: ❌ 불일치 (스코프 밖)
  - 본 레포는 **PyTorch eager + transformers**에서의 hook 데모만 제공
  - vLLM/Triton/DeepSpeed inference 등의 개입 포인트는 문서/코드로 구현하지 않음

---

## 8) 재현성/실행성은 확보되었는가?

- **현재 레포 상태**: ✅ 확보
  - `README.md`: 설치/실행/모델 변경 방법 + 예시 출력 포함
  - `requirements.txt`: 동작 확인된 버전 고정(단, OS/아키텍처에 따라 torch 설치가 달라질 수 있음)
  - macOS에서 CPU `SIGBUS` 이슈를 피하기 위해 MPS 우선 사용 로직 포함
  - `PROMPT_SUITE`/`PRINT_OUTPUT=0`: 두 프롬프트 연속 실행 + 민감 프롬프트 출력 비노출(메타 결과만) 지원

---

## 9) 결론: “보고서대로 구현되었는가?”

- **latent steering / activation hook 데모** 관점에서는: ✅ 대체로 예
- **safety refusal 우회 데모** 관점에서는: ❌ 아니오 (의도적으로 제외)
- **Gemma-3 12B / vLLM 범위** 관점에서는: ❌ 아니오 (스코프 밖)


