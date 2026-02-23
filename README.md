# Local Qwen3-TTS CLI (macOS Intel, Local-Only)

`LOCAL_QWEN3TTS_PLAN_CHECKLIST.md` 기준으로 구현된 로컬 전용 Qwen3-TTS CLI입니다.

## 제약 및 범위
- 로컬 단일 머신 전용
- CLI 전용 (웹 UI 없음)
- 단일 워커 큐 처리 (동시성 1)
- Nearline 처리 지향 (비실시간)
- 원격 GPU/원격 서버 사용 금지

## 디렉터리 표준
실행 시 아래 디렉터리를 자동 생성합니다.
- `runtime/voices/`
- `runtime/jobs/`
- `runtime/output/`
- `logs/`

## 지원 명령 (MVP)
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

## 설치
권장: Python 3.10+ conda 환경

```bash
conda create -n qtts python=3.10 -y
conda activate qtts
pip install -e .
```

필수 패키지:
- `qwen-tts`
- `torch`
- `torchaudio`
- `soundfile`

## 빠른 시작
1. 환경 점검
```bash
qtts doctor
```

2. 음성 클론 생성 (voice-id 생성 또는 재사용)
```bash
qtts clone create \
  --voice-id myvoice \
  --ref-audio ./samples/ref.wav \
  --ref-text "안녕하세요. 음성 클론 기준 문장입니다."
```

3. 단건 합성
```bash
qtts speak \
  --voice-id myvoice \
  --text "로컬 환경에서 합성 테스트를 진행합니다." \
  --out ./out/single.wav
```

4. 배치 합성 (`input.txt` 각 줄 1문장)
```bash
qtts batch \
  --voice-id myvoice \
  --input ./input.txt \
  --out-dir ./out/batch
```

5. 로컬 큐 워커 실행
```bash
qtts serve start --queue runtime/jobs/queue.sqlite3
```

6. 다른 터미널에서 잡 제출/조회/수집
```bash
qtts job submit --voice-id myvoice --text "큐 처리 문장"
qtts job status --job-id <job_id>
qtts job fetch --job-id <job_id> --out ./out/job.wav
```

## 구현 포인트
- 엔진: `qwen_tts.Qwen3TTSModel` API 래핑
  - `create_voice_clone_prompt`
  - `generate_voice_clone`
- 클론 데이터: `runtime/voices/<voice-id>/prompt.json`
  - 레퍼런스 오디오/텍스트 저장
  - 직렬화된 프롬프트 캐시 저장 (재실행 재사용)
- 장문 처리: 문장 단위 자동 분할 + 청크 사이 무음 연결
- 안정화: 청크별 타임아웃/재시도
- 큐: SQLite 영속화
- 단일 워커 보장: 락 파일(`.lock`) + `flock` 사용
- 로그: `logs/qtts.log` JSON lines

## 공통 오류 코드
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

## 운영 가이드
### 재기동 절차
1. `qtts serve start --queue ...` 프로세스 종료 확인
2. 필요 시 락 파일 점검 (`*.lock`)
3. 동일 명령으로 재실행
4. `qtts job status`로 미완료 잡 상태 확인

### 장애 대응
- 모델 로드 실패: `qtts doctor`에서 `import_*` 및 Python 버전 확인
- 메모리/속도 이슈: `--max-new-tokens` 축소, 장문 분할 유지, 배치/큐 처리 사용
- 음성 품질 편차: 레퍼런스 오디오 길이(최소 2.5초 이상), 잡음/겹침 발화 제거

## 성능/품질 기대치
- CPU-only에서는 문장 길이에 따라 지연이 큼
- 실시간 처리 목적이 아닌 Nearline 처리 기준으로 사용

## 테스트
```bash
pytest -q
```

참고: 실제 합성 테스트는 모델/의존성 설치 후 실행해야 합니다.
