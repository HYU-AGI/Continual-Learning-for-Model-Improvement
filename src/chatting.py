#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import argparse, re, sys, glob
import torch
from typing import List, Dict, Tuple

def find_latest_ckpt(root: str, exp_prefix: str, subname: str = "SEQ_full") -> Tuple[str, str]:
    base = os.path.join(root, f"{exp_prefix}-{subname}")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"실행 폴더 없음: {base}")

    runs = [p for p in glob.glob(os.path.join(base, "*")) if os.path.isdir(p)]
    if not runs:
        raise FileNotFoundError(f"실행 폴더가 비어있음: {base}")

    # 타임스탬프 정렬: YYYY-MM-DD-HH-MM-SS 포맷이라 역순 정렬로 최신
    runs.sort(reverse=True)
    pat = re.compile(r"last_ckpt_task(\d+)\.pth$")

    for run in runs:
        ckpts = glob.glob(os.path.join(run, "last_ckpt_task*.pth"))
        if not ckpts:
            continue
        # 가장 큰 task id 선택
        def key_fn(p):
            m = pat.search(os.path.basename(p))
            return int(m.group(1)) if m else -1
        ckpts.sort(key=key_fn, reverse=True)
        best = ckpts[0]
        if pat.search(os.path.basename(best)):
            return run, best

    raise FileNotFoundError(f"ckpt를 찾지 못함: {base}/<run>/last_ckpt_task*.pth")

def load_tokenizer(model_or_id: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_or_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    return tok

def unwrap_generate_module(obj):
    if hasattr(obj, "generate"):
        return obj
    if hasattr(obj, "module") and hasattr(obj.module, "generate"):
        return obj.module
    if hasattr(obj, "model") and hasattr(obj.model, "generate"):
        return obj.model
    return None

def load_from_ckpt(ckpt_path: str, base_model: str | None, device: str, dtype: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    obj = None
    if isinstance(ckpt, dict) and "model" in ckpt:
        obj = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        obj = ckpt["state_dict"]
    else:
        obj = ckpt

    gen = unwrap_generate_module(obj)
    if gen is not None:
        gen.to(device)
        gen.eval()
        return gen, "object"

    if base_model is None:
        raise RuntimeError("state_dict만 발견됨. --base_model 을 지정해야 로드 가능")
    from transformers import AutoModelForCausalLM
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype, device_map="auto")
    missing, unexpected = model.load_state_dict(obj, strict=False)
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected}")
    if missing:
        print(f"[warn] missing keys: {missing}")
    model.eval()
    return model, "state_dict"

def render_chat(tokenizer, messages: List[Dict[str,str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    parts = []
    for m in messages:
        if m["role"] == "system":
            parts.append(f"[System] {m['content']}")
        elif m["role"] == "user":
            parts.append(f"User: {m['content']}")
        elif m["role"] == "assistant":
            parts.append(f"Assistant: {m['content']}")
    parts.append("Assistant:")
    return "\n".join(parts)

@torch.inference_mode()
def chat_once(model, tok, history, max_new_tokens=256, temperature=0.7, top_p=0.9, repetition_penalty=1.05):
    prompt = render_chat(tok, history)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    gen = model.generate(
        **inputs,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    out = tok.decode(gen[0], skip_special_tokens=True)
    return out.split("Assistant:")[-1].strip() if "Assistant:" in out else out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda_device", type=str, default=None, help="CUDA device to use (e.g., 0 or 0,1)")
    ap.add_argument("--exp_prefix", required=True, help="예: my-exp")
    ap.add_argument("--root", default="experiments")
    ap.add_argument("--subname", default="SEQ_full")
    ap.add_argument("--base_model", default="EleutherAI/pythia-410m-deduped", help="토크나이저/병합용 베이스")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16","bfloat16","float32"])
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--dry_run", action="store_true", help="선택된 경로만 출력하고 종료")
    args = ap.parse_args()

    # CUDA device 설정
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    run_dir, ckpt = find_latest_ckpt(args.root, args.exp_prefix, args.subname)
    print(f"[info] run_dir = {run_dir}")
    print(f"[info] ckpt    = {ckpt}")

    if args.dry_run:
        return

    tok = load_tokenizer(args.base_model)
    model, how = load_from_ckpt(ckpt, base_model=args.base_model, device=args.device, dtype=args.dtype)
    print(f"[info] checkpoint loaded via: {how}")

    history = [dict(role="system", content=args.system)]
    if args.once:
        try:
            user_msg = input("User: ").strip()
        except EOFError:
            return
        history.append(dict(role="user", content=user_msg))
        ans = chat_once(model, tok, history,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty)
        print(f"\nAssistant: {ans}\n")
        return

    print("Enter로 전송. /exit 종료, /reset 초기화")
    while True:
        try:
            user_msg = input("User: ").strip()
        except EOFError:
            break
        if user_msg == "/exit":
            break
        if user_msg == "/reset":
            history = [dict(role="system", content=args.system)]
            print("History cleared.")
            continue
        if not user_msg:
            continue
        history.append(dict(role="user", content=user_msg))
        ans = chat_once(model, tok, history,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty)
        history.append(dict(role="assistant", content=ans))
        print(f"Assistant: {ans}\n")

if __name__ == "__main__":
    main()
