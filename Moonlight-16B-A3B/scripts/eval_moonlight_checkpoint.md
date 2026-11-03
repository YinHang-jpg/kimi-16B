# Evaluating released Moonlight checkpoints (official metrics)

MoonshotAI **did not open-source** an evaluation suite inside
[MoonshotAI/Moonlight](https://github.com/MoonshotAI/Moonlight).  
Paper tables report: MMLU, MMLU-pro, BBH, TriviaQA, HumanEval, MBPP, GSM8K,
MATH, CMath, C-Eval, CMMLU.

## Recommended community tool

[EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

Example (after installing `lm_eval` and task data; use your local weight path):

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source .venv/bin/activate

# Illustrative; exact task names depend on lm-eval version.
lm_eval --model hf \
  --model_args pretrained=./,trust_remote_code=True,dtype=bfloat16 \
  --tasks mmlu,gsm8k,humaneval,ceval-valid,cmmlu \
  --device npu:0 \
  --batch_size 1
```

Notes:

1. This evaluates the **released HF checkpoint**, not a model you re-trained
   from scratch on 5.7T tokens (that pipeline is not open-sourced).
2. On air-gapped Ascend hosts, `lm_eval` and task datasets must be offline-
   vendored the same way as other wheels.
3. Do not treat toy_train loss as equivalent to the paper’s benchmark table.
