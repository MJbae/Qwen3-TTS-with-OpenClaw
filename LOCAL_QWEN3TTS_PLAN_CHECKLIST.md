# Qwen3-TTS 로컬 전용 구현 계획 및 체크리스트

작성일: 2026-02-23  
대상 환경: macOS Intel (로컬 단일 머신)  
제약사항: **원격 GPU/원격 서버 사용 금지**, **CLI 인터페이스만 제공**

## 1) 목표와 범위

### 목표
- 로컬 환경에서 음성 클론(`voice-id`) 생성
- 생성된 화자로 CLI 기반 TTS 합성 제공
- 단건 합성 + 배치 합성 + 로컬 큐 기반 서빙(단일 워커)

### 비목표
- 실시간(저지연) 응답 보장
- 다중 동시 요청 고성능 처리
- 웹 UI 제공

## 2) 구현 전략 (현실성 우선)

- 모델 우선순위: `Qwen3-TTS-1.7B` CPU-local 기본, `0.6B`는 fallback
- 추론 디바이스: CPU-only 기본
- 운영 형태: 단일 프로세스 + 요청 큐(파일/SQLite)
- 결과 전달: 파일 출력(WAV) + 잡 상태 조회
- 품질/속도 트레이드오프를 문서화하고 Nearline 처리로 정의
- fallback 정책: 성능/메모리 이슈 시 `--model-id Qwen/Qwen3-TTS-12Hz-0.6B-Base`로 즉시 전환

## 3) 단계별 작업 계획

## Phase 0. 사전 확정 (반드시 선행)
- 요구사항 고정: 로컬 전용, CLI 전용, 단일 워커
- 디렉터리 표준:
  - `runtime/voices/`
  - `runtime/jobs/`
  - `runtime/output/`
  - `logs/`
- 산출물: `README.md`에 제약/범위 명시

완료 기준:
- 범위/비목표가 문서로 고정되고 합의됨

## Phase 1. 환경 및 설치 게이트 (Go/No-Go)
- Python 런타임 정리(권장: conda env, Python 3.10+)
- `qwen-tts`, `torch`, `torchaudio` 설치
- CPU-only 모델 로드 스모크 테스트

완료 기준:
- `python -c "import qwen_tts"` 성공
- 샘플 1문장 TTS 파일 생성 성공
- 실패 시 대응:
  - 의존성 버전 조합 고정
  - 실행 불가 시 즉시 범위 재조정(클론 제외/합성만)

## Phase 2. 음성 클론 파이프라인
- CLI: `clone create`
- 입력: `--voice-id --ref-audio --ref-text`
- 저장: `runtime/voices/<voice-id>/prompt.json` 및 메타데이터

완료 기준:
- 같은 `voice-id`로 재실행 시 프롬프트 재사용 가능
- 잘못된 입력(짧은 오디오/파일 없음) 오류 처리

## Phase 3. TTS 합성 CLI
- CLI: `speak`
- 입력: `--voice-id --text --out`
- 옵션: `--language --seed --max-new-tokens`

완료 기준:
- 단건 합성 성공
- 실패 시 오류 코드/메시지 명확화(`VOICE_NOT_FOUND`, `MODEL_LOAD_FAIL` 등)

## Phase 4. 배치 합성 및 로컬 서빙
- CLI: `batch`, `serve start`, `job submit`, `job status`, `job fetch`
- 내부 큐: SQLite 또는 파일 큐 중 1개 선택
- 단일 워커 정책 강제(동시성 1)

완료 기준:
- 10문장 배치 처리 성공
- 중간 실패 건 재시도/재개 가능

## Phase 5. 품질, 안정성, 운영 문서
- 장문 입력 자동 분할(문장 단위)
- 타임아웃/재시도/로그 정비
- 운영 문서: 실행법, 장애 대응, 성능 기대치

완료 기준:
- 재부팅 후 서비스 재기동 절차 확인
- 로그만으로 실패 원인 분류 가능

## 4) CLI 명세 (MVP)

```bash
qtts doctor
qtts clone create --voice-id <id> --ref-audio <wav> --ref-text "<text>"
qtts speak --voice-id <id> --text "<text>" --out <wav>
qtts batch --voice-id <id> --input <txt> --out-dir <dir>
qtts voices list
qtts voices delete --voice-id <id>
qtts serve start --queue <sqlite_or_dir>
qtts job submit --voice-id <id> --text "<text>"
qtts job status --job-id <id>
qtts job fetch --job-id <id> --out <wav>
```

## 5) 작업 체크리스트

### A. 환경 준비
- [ ] conda env 생성 (Python 3.10+)
- [ ] `pip install qwen-tts torch torchaudio` 완료
- [ ] 모델 다운로드 위치 결정(로컬 디스크 경로 고정)
- [ ] `qtts doctor`에서 필수 항목 통과

### B. 최소 기능 검증
- [ ] CPU-only 모델 로드 성공
- [ ] 기본 TTS 1문장 WAV 출력 성공
- [ ] 음성 클론 1회 생성 성공
- [ ] 생성된 `voice-id` 재사용 합성 성공
- [ ] fallback 모델(`0.6B`) 오버라이드 실행 확인

### C. CLI 기능 구현
- [ ] `clone create` 구현
- [ ] `speak` 구현
- [ ] `batch` 구현
- [ ] `voices list/delete` 구현
- [ ] 공통 에러 코드 체계 구현

### D. 로컬 서빙 구현
- [ ] `serve start` 구현
- [ ] `job submit/status/fetch` 구현
- [ ] 큐 영속화(SQLite 또는 파일) 구현
- [ ] 단일 워커 락(lock) 적용

### E. 안정화
- [ ] 타임아웃/재시도 정책 적용
- [ ] 장문 자동 분할 적용
- [ ] 로그 구조화(JSON lines 권장)
- [ ] 실패 복구 시나리오 테스트

### F. 검증
- [ ] 스모크 테스트: 클론 1회 + 합성 3문장
- [ ] 배치 테스트: 10문장 성공
- [ ] 회귀 테스트: 동일 입력 재실행 안정성 확인
- [ ] 성능 기록: 평균 처리 시간/문장 길이별 지연

### G. 문서화
- [ ] 설치 가이드
- [ ] 실행 가이드(단건/배치/서빙)
- [ ] 트러블슈팅(메모리 부족, 모델 로드 실패 등)
- [ ] 제약사항(비실시간, 단일 워커) 명시

## 6) 리스크와 대응

- 리스크: 1.7B CPU-only 속도/메모리 한계
  - 대응: Nearline 정책, 배치 처리, 장문 분할, 필요 시 0.6B fallback 즉시 전환
- 리스크: macOS Intel 의존성 충돌
  - 대응: 버전 매트릭스 고정, 환경 재현 스크립트 제공
- 리스크: 클론 품질 편차
  - 대응: 레퍼런스 오디오 가이드(잡음/길이/발화 명료도) 문서화

## 7) 수용 기준 (Acceptance Criteria)

- 로컬 단일 머신에서 `voice-id` 생성 가능
- 동일 화자로 10문장 배치 합성 성공
- CLI 오류가 원인별로 구분되어 사용자 조치 가능
- 문서 기준으로 신규 환경 재현 설치 가능
