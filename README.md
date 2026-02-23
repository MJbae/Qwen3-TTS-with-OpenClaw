# Local Qwen3-TTS CLI

로컬 단일 머신에서 Qwen3-TTS 음성 클론과 TTS 합성을 수행하는 CLI 프로젝트입니다.
이 문서는 **`/Users/bot/tts-server` 머신/경로 전용 운영 문서**입니다.

- 로컬 전용
- CLI 전용
- 단일 워커 큐(동시성 1)
- Nearline 처리(비실시간)

## 1. 주요 기능
- `voice-id` 생성/재사용 (`clone create`)
- 단건 합성 (`speak`)
- 배치 합성 (`batch`)
- 음성 목록/삭제 (`voices list`, `voices delete`)
- 로컬 큐 워커 (`serve start`, `job submit/status/fetch`)
- 상태 점검 (`doctor`)

## 2. 디렉터리 구조
실행 시 자동 생성/사용됩니다.

- `runtime/voices/`
- `runtime/jobs/`
- `runtime/output/`
- `logs/`

## 3. 요구사항
- macOS Intel (검증 환경)
- Python 3.10+
- conda (이 문서 기준 운영 환경)

## 4. 설치
### 4.1 권장: 프로젝트 전용 env 생성
```bash
conda create -y -p /Users/bot/tts-server/.conda/qtts python=3.10
conda activate /Users/bot/tts-server/.conda/qtts
```

### 4.2 의존성 설치 (이 머신 검증 순서)
`pyproject.toml`에는 일반 의존성 집합이 정의되어 있지만, 이 머신에서는 `qwen-tts` 의존 체인
(`librosa/numba/llvmlite`) 설치 충돌을 피하기 위해 아래 순서를 사용합니다.

```bash
pip install numpy torch==2.2.2 torchaudio==2.2.2 soundfile pytest transformers==4.57.3 accelerate==1.12.0
pip install librosa==0.10.2.post1 numba==0.59.1 llvmlite==0.42.0 scipy scikit-learn joblib pooch soxr audioread lazy_loader msgpack
pip install onnxruntime einops sox qwen-tts --no-deps
pip install -e . --no-deps
conda install -y -c conda-forge sox
```

### 4.3 실행 PATH 설정
`sox` 바이너리와 `qtts` 엔트리포인트를 우선 사용하도록 PATH를 설정합니다.

```bash
export PATH=/Users/bot/tts-server/.conda/qtts/bin:$PATH
```

## 5. 빠른 시작
### 5.1 상태 점검
```bash
qtts doctor
```

### 5.2 스모크 합성 점검(선택)
`doctor`에서 실제 합성까지 확인할 때 사용합니다.

```bash
qtts doctor --run-smoke --voice-id myvoice --smoke-text "Doctor smoke test sentence."
```

### 5.3 음성 클론 생성
```bash
qtts clone create \
  --voice-id myvoice \
  --ref-audio ./samples/ref.wav \
  --ref-text "This is a reference sentence for voice clone."
```

### 5.4 단건 합성
```bash
qtts speak \
  --voice-id myvoice \
  --text "This is a synthesis test." \
  --out ./runtime/output/myvoice_test.wav
```

### 5.5 배치 합성
`input.txt`는 줄당 1문장 형식입니다.

```bash
qtts batch \
  --voice-id myvoice \
  --input ./input.txt \
  --out-dir ./runtime/output/batch
```

## 6. 큐 모드 사용법
### 6.1 워커 시작
```bash
qtts serve start --queue runtime/jobs/queue.sqlite3
```

### 6.2 잡 제출/조회/결과 수집
```bash
qtts job submit --queue runtime/jobs/queue.sqlite3 --voice-id myvoice --text "Queue test"
qtts job status --queue runtime/jobs/queue.sqlite3 --job-id <job_id>
qtts job fetch --queue runtime/jobs/queue.sqlite3 --job-id <job_id> --out ./runtime/output/job.wav
```

## 7. 전체 명령
```bash
qtts doctor
qtts doctor --run-smoke --voice-id <id> --smoke-text "<text>"
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

## 8. 공통 오류 코드
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

## 9. 테스트
```bash
python -m pytest -q
```

## 10. 트러블슈팅
### 10.1 `SoX could not be found!`
- 원인: `sox` 실행 파일이 PATH에 없음
- 해결:
```bash
conda install -y -c conda-forge sox
```

### 10.2 `OMP: Error #179: Can't open SHM2`
- 원인: 일부 제한된 실행 환경(샌드박스/격리 실행)에서 OpenMP 공유메모리 접근 실패
- 해결: 일반 로컬 터미널(권한 제한 없는 세션)에서 실행

### 10.3 `flash-attn is not installed` 경고
- 의미: 성능 경고이며 CPU 기본 실행에는 치명적이지 않음

## 11. 운영 메모
- `runtime/voices/<voice-id>/prompt.json`에 voice metadata/prompt cache 저장
- 로그 파일: `logs/qtts.log` (JSON lines)
- 배치/큐는 실패 항목 재시도 가능 정책을 적용
