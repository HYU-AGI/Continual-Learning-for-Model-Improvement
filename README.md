# Continual-Learning-for-Model-Improvement
## 프롬프팅 및 RAG 기법과 독립적으로 AI 모델 자체의 지속적인 성능 향상을 위한 Continual Learning 기법
### 💡 예시
![image](./img/example.png)

## ⚙️ Requirements
To install requirements:
```
conda create -n ilvenv python=3.11.10
conda activate ilvenv
pip install -r requirements.txt
```

## 💻 Running Continual-Learning
### Step 1. 준비된 데이터셋으로 Continual Learning 진행
```
./scripts/run.sh
```

### Step 2. 학습된 모델 테스트
```
./scripts/chatting.sh
```

## 🧪 예시 데모
아래는 실제 사용 흐름을 보기 좋게 정리한 터미널 세션 예시다. 명령 프롬프트 기호와 시스템 메시지를 포함해 입출력이 한눈에 들어오도록 구성했다.

```text
(ilvenv) root@82c32631fb72:/workspace/Continual-Learning-for-Model-Improvement$ ./scripts/chatting.sh

───────────────────────────────────────────────────────────────────────────────

user      > Michelle Obama married Barack Obama in 1992. The relationship between Michelle Obama and Barack Obama is
assistant > person related: spouse

user      > Google was founded by Larry Page and Sergey Brin at Stanford. The relationship between Google and Larry Page is
assistant > organization related: founded by
```

## 🧠 Continual-Learning 작동 원리

이 프로젝트의 Continual Learning은 2단계 학습 과정을 통해 모델의 점진적 학습을 구현합니다:

### 1️⃣ 초기 Backbone 학습 단계
- 첫 번째 학습 단계에서는 모델의 backbone 부분만 학습됩니다
- 이는 기본적인 특징 추출(feature extraction)과 표현 학습(representation learning)을 위한 단계입니다
- backbone은 모델의 핵심 아키텍처로, 입력 데이터의 주요 특징을 학습합니다

### 2️⃣ Language Model Head 학습 단계
- 두 번째 단계에서는 이전에 학습된 backbone은 고정(freeze)됩니다
- 이 단계에서는 Language Model Head만 학습이 진행됩니다
- LM Head는 이전 단계에서 학습된 특징을 바탕으로 새로운 태스크에 적응합니다
- backbone을 고정함으로써 이전 학습된 지식을 보존하면서 새로운 지식을 습득할 수 있습니다

### 💡 장점
- 파국적 망각(Catastrophic Forgetting) 방지
- 효율적인 학습: 전체 모델이 아닌 일부분만 학습하여 계산 효율성 향상
- 안정적인 성능: 기존 지식을 유지하면서 새로운 태스크 학습 가능

`scripts/run.sh`를 실행하면 위 과정이 자동으로 순차적으로 진행되며, 각 단계별 학습 진행 상황은 wandb를 통해 모니터링할 수 있습니다.

## Reference

This project builds on:
- [ACL 2024] A Codebase for Incremental Learning with Large Language Models
  Code: https://github.com/zzz47zzz/codebase-for-incremental-learning-with-llm
