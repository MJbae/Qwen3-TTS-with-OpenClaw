# Local Qwen3-TTS CLI

로컬 단일 머신 환경에서 Qwen3-TTS 기반 음성 클론과 합성을 수행하는 CLI 도구입니다.

## 개요
- 로컬 전용 TTS/Voice Clone 워크플로
- 웹 UI 없이 CLI 명령만 제공
- 단건 합성, 배치 합성, SQLite 기반 로컬 큐 지원
- 단일 워커 정책(동시성 1)

## 범위 및 제약
- 실시간(저지연) 응답 보장은 목표가 아닙니다.
- 고성능 다중 동시 처리 대신 Nearline 처리에 초점을 둡니다.
- 원격 서버/원격 GPU 없이 로컬 실행을 전제로 합니다.

## 주요 기능
- `clone create`: `voice-id` 생성/재사용
- `speak`: 단건 합성
- `batch`: 텍스트 파일(줄당 1문장) 배치 합성
- `voices list/delete`: 화자 목록/삭제
- `serve start` + `job submit/status/fetch`: 로컬 큐 워커/잡 처리
- `doctor`: 의존성 및 런타임 상태 점검

## 요구사항
- Python 3.10+
- `sox` 실행 파일이 PATH에 있어야 함
- (권장) conda 또는 venv

## 설치

### 방법 1) 일반 설치 (권장 시작점)
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

또는 conda:
```bash
conda create -n qtts python=3.10 -y
conda activate qtts
pip install --upgrade pip
pip install -e .[dev]
```

### 방법 2) qwen-tts 의존성 충돌 시(대안)
일부 환경에서 `qwen-tts` 설치 시 `librosa/numba/llvmlite` 충돌이 날 수 있습니다.
그 경우 아래 순서를 사용하세요.

```bash
pip install numpy torch==2.2.2 torchaudio==2.2.2 soundfile pytest transformers==4.57.3 accelerate==1.12.0
pip install librosa==0.10.2.post1 numba==0.59.1 llvmlite==0.42.0 scipy scikit-learn joblib pooch soxr audioread lazy_loader msgpack
pip install onnxruntime einops sox qwen-tts --no-deps
pip install -e . --no-deps
```

### SoX 설치
- macOS(Homebrew):
```bash
brew install sox
```
- conda:
```bash
conda install -c conda-forge sox
```
- Ubuntu/Debian:
```bash
sudo apt-get update && sudo apt-get install -y sox libsox-fmt-all
```

## 빠른 시작

### 1) 환경 점검
```bash
qtts doctor
```

선택: 실제 합성 스모크 테스트
```bash
qtts doctor --run-smoke --voice-id myvoice --smoke-text "Doctor smoke test sentence."
```

### 2) 음성 클론 생성
```bash
qtts clone create \
  --voice-id myvoice \
  --ref-audio ./samples/ref.wav \
  --ref-text "This is a reference sentence for voice clone."
```

### 3) 단건 합성
```bash
qtts speak \
  --voice-id myvoice \
  --text "This is a synthesis test." \
  --out ./runtime/output/myvoice_test.wav
```

### 4) 배치 합성
```bash
qtts batch \
  --voice-id myvoice \
  --input ./input.txt \
  --out-dir ./runtime/output/batch
```

## 큐 모드

### 워커 시작
```bash
qtts serve start --queue runtime/jobs/queue.sqlite3
```

### 잡 제출/조회/결과 수집
```bash
qtts job submit --queue runtime/jobs/queue.sqlite3 --voice-id myvoice --text "Queue test"
qtts job status --queue runtime/jobs/queue.sqlite3 --job-id <job_id>
qtts job fetch --queue runtime/jobs/queue.sqlite3 --job-id <job_id> --out ./runtime/output/job.wav
```

## CLI 명령 목록
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

## 자주 쓰는 옵션
공통 합성 명령(`doctor --run-smoke`, `speak`, `batch`, `serve start`)에서 사용:
- `--language`
- `--seed`
- `--max-new-tokens`
- `--timeout-sec`
- `--retries`
- `--split-max-chars`

전역 옵션:
- `--runtime-root`: `runtime` 기본 경로 변경
- `--model-id`: 기본 `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
- `--device`: 기본 `cpu`
- `--attn-implementation`
- `--verbose`

## 런타임 경로
실행 시 아래 디렉터리를 자동 생성/사용합니다.
- `runtime/voices/`
- `runtime/jobs/`
- `runtime/output/`
- `logs/`

참고: `--runtime-root`는 `runtime/*` 경로에 적용되며, 로그는 기본적으로 `logs/qtts.log`를 사용합니다.

## 오류 코드
- `INVALID_INPUT`
- `AUDIO_VALIDATION_FAIL`
- `VOICE_NOT_FOUND`
- `VOICE_ALREADY_EXISTS`
- `MODEL_LOAD_FAIL`
- `SYNTHESIS_FAIL`
- `TIMEOUT`
- `IMPORT_FAIL`
- `QUEUE_LOCKED`
- `JOB_NOT_FOUND`
- `JOB_NOT_READY`
- `INTERNAL_ERROR`

## 테스트
```bash
python -m pytest -q
```

## 트러블슈팅

### `SoX could not be found!`
- 원인: `sox` 실행 파일이 PATH에 없음
- 해결: 위 설치 섹션의 SoX 설치 후 재실행

### `OMP: Error #179: Can't open SHM2`
- 원인: 일부 샌드박스/격리 환경에서 OpenMP 공유메모리 접근이 차단됨
- 해결: 일반 로컬 터미널 세션에서 실행

### `flash-attn is not installed` 경고
- 의미: 성능 관련 경고입니다. CPU 기본 동작에는 필수는 아닙니다.
