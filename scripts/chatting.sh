# 최신 SEQ_full ckpt로 한 번만 응답
python chatting.py \
  --exp_prefix default-exp \
  --base_model EleutherAI/pythia-410m-deduped \
  --once

# 선택된 경로만 확인하고 싶다면
python chatting.py --exp_prefix {your-experiment-name} --dry_run