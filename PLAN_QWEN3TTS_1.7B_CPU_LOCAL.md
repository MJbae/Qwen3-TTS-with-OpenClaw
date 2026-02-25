# Qwen3-TTS 1.7B CPU 로컬 전환 작업 계획

- 작성일: 2026-02-25
- 대상 디렉토리: `/Users/bot/tts-server-1.7`
- 기준 코드베이스: `/Users/bot/tts-server` (현재 0.6B 동작본)
- 실행 제약: GPU 없음 (CPU-only)

## 1. 목표

1. 기존 `Qwen/Qwen3-TTS-12Hz-0.6B-Base` 기본 구성을 `Qwen/Qwen3-TTS-12Hz-1.7B-Base` 기준으로 전환한다.
2. GPU 없이 동일 로컬 환경에서 `doctor/speak/clone/batch/queue` 경로가 실제로 동작하도록 세팅한다.
3. 느려진 추론 시간과 메모리 사용량을 감안해 안전한 기본값(타임아웃/청크/스레드)과 검증 게이트를 추가한다.

## 2. 공식 근거 (Primary Sources)

1. Qwen3-TTS 공식 저장소의 모델 목록에 `1.7B`(CustomVoice/VoiceDesign/Base)와 `0.6B`가 함께 제시됨.  
   - https://github.com/QwenLM/Qwen3-TTS
2. 공식 예제는 `device_map="cuda:0"`, `dtype=torch.bfloat16`, `attn_implementation="flash_attention_2"` 조합을 사용함.  
   - https://github.com/QwenLM/Qwen3-TTS
3. `qwen_tts.Qwen3TTSModel.from_pretrained(..., **kwargs)`는 kwargs를 `AutoModel.from_pretrained`로 전달함(로컬 코드 점검).  
   - 로컬 site-packages: `qwen_tts/inference/qwen3_tts_model.py`
4. Hugging Face 모델 파일 크기(2026-02-25 조회):
   - `Qwen3-TTS-12Hz-1.7B-Base` `model.safetensors`: `3,857,413,744` bytes
   - `Qwen3-TTS-12Hz-0.6B-Base` `model.safetensors`: `1,829,344,272` bytes
   - `speech_tokenizer/model.safetensors`: 두 모델 공통 `682,293,092` bytes
   - https://huggingface.co/api/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base?blobs=true
   - https://huggingface.co/api/models/Qwen/Qwen3-TTS-12Hz-0.6B-Base?blobs=true

## 3. CPU-only 제약과 운영 원칙

1. 공식 문서는 GPU 가속 중심이다. CPU-only는 가능하더라도 체감 지연이 크게 증가할 수 있다.  
   - 위 공식 예제가 CUDA/flash-attn을 사용한다는 근거에서 도출한 추론.
2. CPU 경로 기본 전략:
   - `device=cpu`
   - `dtype=float32` 유지 (호환성 우선)
   - `attn_implementation=eager` 기본화
   - 청크 길이 축소 + 타임아웃 상향
3. 안정성 우선 전략:
   - 1차 목표는 “실패 없이 완주(무 OOM/무 크래시)”
   - 2차 목표는 “속도 최적화(스레드/청크 튜닝)”

## 4. 서브 에이전트 기반 분할 계획

### SA-1: Upstream/환경 잠금

- 책임: 1.7B 전환에 필요한 의존성/실행환경 명세 고정
- 작업:
  1. `/Users/bot/tts-server`를 기준으로 `/Users/bot/tts-server-1.7` 초기 스캐폴드 구성
  2. Python/패키지 버전 잠금안 작성 (`qwen-tts`, `torch`, `transformers`)
  3. 모델 사전 다운로드 절차(HF CLI) 문서화
- 산출물:
  - `README.md` 환경 섹션 초안
  - `requirements-lock` 또는 설치 절차 문서
- 완료 기준:
  - 새 환경에서 `import torch, qwen_tts` 성공
  - 1.7B 모델 다운로드/캐시 경로 확인 가능

### SA-2: 모델 로더/런타임 코드 전환

- 책임: 1.7B CPU 안정 로딩과 실패 시 메시지 개선
- 소유 파일(예상):
  - `src/qtts/constants.py`
  - `src/qtts/engine.py`
- 작업:
  1. 기본 모델 ID를 `Qwen/Qwen3-TTS-12Hz-1.7B-Base`로 변경
  2. CPU 전용 안전 옵션 추가:
     - `low_cpu_mem_usage` 전달
     - CPU 스레드 제한(`torch.set_num_threads`) 옵션화
  3. OOM/모델 로드 실패 시 액션 가능한 에러 문구 추가
- 완료 기준:
  - 1.7B CPU 로드 성공
  - 실패 시 원인 분류(메모리/의존성/모델ID)가 로그에 남음

### SA-3: CLI/Doctor/서비스 경로 튜닝

- 책임: 실제 사용자 경로의 기본값/검증 게이트 조정
- 소유 파일(예상):
  - `src/qtts/cli.py`
  - `src/qtts/service.py`
  - `src/qtts/text_utils.py` (필요 시)
- 작업:
  1. CLI 기본값 재설정:
     - timeout 상향
     - split-max-chars 하향
  2. 새 플래그 추가:
     - `--cpu-threads`
     - `--low-cpu-mem`
  3. `doctor --run-smoke`를 1.7B CPU 경로로 검증하도록 보강
- 완료 기준:
  - `doctor --run-smoke` 통과
  - `speak/batch/clone` 명령이 1.7B 기본 설정으로 정상 동작

### SA-4: 성능 계측/회귀 테스트/문서화

- 책임: CPU 환경에서의 운영 가능성 판단과 릴리스 기준 수립
- 소유 파일(예상):
  - `tests/*` (신규 테스트)
  - `docs/*` 또는 `README.md`
  - `scripts/*` (벤치 스크립트)
- 작업:
  1. 기능 게이트:
     - speak 1건
     - batch 1건
     - queue submit/status/fetch 1사이클
  2. 성능 게이트(초기 권장값):
     - OOM 없이 완료
     - 피크 RSS가 시스템 RAM(32GB) 내 안전 여유 확보
     - 단문/중문 추론 시간 기록 및 기준선 확정
  3. 운영 문서:
     - CPU-only 한계/튜닝 팁/0.6B fallback 절차
- 완료 기준:
  - 기능 게이트 전부 통과
  - 측정 결과가 문서화됨

## 5. 구현 순서 (권장)

1. Phase 0: 디렉토리 스캐폴드 + 환경 잠금 (SA-1)
2. Phase 1: 모델 로더 전환 (SA-2)
3. Phase 2: CLI/doctor/서비스 기본값 조정 (SA-3)
4. Phase 3: 기능/성능 게이트 및 문서화 (SA-4)

## 6. 상세 체크리스트

1. 프로젝트 준비
   - `/Users/bot/tts-server-1.7`에 코드베이스 구성
   - `.conda/qtts` 또는 동일 격리환경 생성
2. 기본 모델 전환
   - 상수/기본값을 1.7B로 변경
   - CLI 오버라이드(`--model-id`)는 유지
3. CPU 안정화
   - 로드 옵션(`low_cpu_mem_usage`) 전달
   - 스레드/타임아웃/청크 기본값 튜닝
4. 검증
   - `doctor --run-smoke`
   - `clone create` + `speak`
   - `batch` + `serve/job` 경로
5. 문서/릴리스
   - CPU-only 운영 가이드
   - 장애 대응(0.6B fallback) 가이드

## 7. 수용 기준 (Acceptance Criteria)

1. `doctor --run-smoke`가 1.7B 기본값으로 성공한다.
2. `speak`, `clone`, `batch`, `queue` 핵심 명령이 CPU-only에서 실패 없이 완료된다.
3. 모델 로드/추론 실패 시 원인별 안내 메시지가 제공된다.
4. README에 CPU-only 한계와 fallback 절차가 명시된다.

## 8. 롤백 계획

1. 항상 `--model-id Qwen/Qwen3-TTS-12Hz-0.6B-Base` 즉시 전환 가능 상태 유지
2. 1.7B 전환 커밋은 단계별로 분리하여 부분 롤백 가능하게 유지
3. 성능/안정성 기준 미달 시 기본 모델을 0.6B로 되돌리고 원인 분석 후 재시도

## 9. 착수 시 첫 실행 명령(초안)

```bash
cd /Users/bot/tts-server-1.7
# (코드 스캐폴드 구성 후)
./scripts/qtts-run.sh doctor
./scripts/qtts-run.sh doctor --run-smoke --voice-id smoke
./scripts/qtts-run.sh speak --text "CPU 1.7B smoke test" --voice-id smoke --out runtime/out/smoke.wav
```

---

이 문서는 “GPU 없는 동일 로컬 환경에서 1.7B로 전환”을 위한 구현 기준 문서이며, 실제 구현은 위 Phase 순서대로 진행한다.
