# 官方能力核对与本仓库补齐说明（务必先读）

对照上游：[MoonshotAI/Moonlight](https://github.com/MoonshotAI/Moonlight) 与论文
[Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)。

## 你的四个问题（直接回答）

### 1. 当前是否有「完整训练脚本」？

**没有「复现 Moonlight-16B（5.7T tokens）全量训练」的官方脚本。**

官方开源的训练入口只有：

- `examples/toy_train.py`
- 训练的是 **小型 Qwen2 结构**（默认 `hidden_size=896/1024`，12 层），**不是** 16B MoE Moonlight
- README 中的 Megatron 全量训练标注为后续/社区方向；论文写 Distributed Muon 会向 Megatron-LM 提 PR，**Moonlight 仓库内并未发布 16B MoE 训练工程**

本仓库已原样 vendor：

- `third_party/MoonshotAI-Moonlight/examples/toy_train.py`

并为 NPU 做了**最小设备适配**副本：

- `examples/toy_train_npu.py`（改动点见文件头 PATCHES）

### 2. 是否有与官方宣称一致的完整数据集？

**没有。**

| 官方宣称 | 开源情况 |
|---------|---------|
| Moonlight 用 **5.7T tokens** 训练 | **未开源**该语料 |
| toy 示例用 `openwebtext-100k`（`Elriggs/openwebtext-100k`） | 需从 Hugging Face 拉取，**不在**权重目录里 |

因此：**无法在本机「完整重跑一遍与官方 Moonlight 相同数据、相同规模的训练」来对齐 16B 精度表。**  
可复现的是官方开源的 **toy 规模 Muon 训练**。

数据准备（有外网/可中转时）：

```bash
python scripts/prefetch_official_toy_data.py
```

会把 toy 脚本依赖的数据集与 `Qwen/Qwen2.5-0.5B` tokenizer 缓存到本地目录，供离线机使用。

### 3. 是否有评测精度套件？

**Moonlight 官方仓库未开源评测代码。**  
论文表格指标（MMLU、HumanEval、GSM8K、C-Eval 等）是结果汇报，不是可运行套件。

社区常用做法是用 [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 评已发布 checkpoint。  
本仓库提供：

- `scripts/eval_moonlight_checkpoint.md`：按论文表格列出任务与建议命令（需自行安装 lm-eval；内网需离线 wheel）

这只能评测 **已发布的 Moonlight 权重**，不能替代「从头训完再评」——因为全量训练链未开源。

### 4. 当前能否在 NPU 上运行？

| 能力 | 状态 |
|------|------|
| 环境 smoke / 加载权重 | 可以（已验证路径） |
| 官方 toy Muon 训练（小模型） | 可以：用 `examples/toy_train_npu.py` |
| 16B MoE + 5.7T 全量复现 | **不能**（缺官方脚本与数据；2 卡也不现实） |
| 论文级 benchmark 全表 | 需另装 lm-eval + 各任务数据；非 Moonlight 官方附带 |

## 对你「对比自研 Muon」目标的可行含义

上级要的「和 Kimi/Moonlight 同精度」若指 **16B + 5.7T 全量对齐**，在**仅使用官方开源材料**的前提下 **做不到**，缺口在官方，不在你的 NPU。

在官方开源范围内，合规对比应改为：

1. 用官方 `toy_train` 里的 **Muon 实现**跑通小模型训练（基线）
2. 把优化器换成你们的 Muon，保持其余脚本/数据/超参不变
3. 对比 loss 曲线与（可选）下游小评测

插件点：`examples/toy_train_npu.py` / 上游 `Muon` 类。

## 一键跑官方 toy（NPU）

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
cd Moonlight-16B-A3B
source .venv/bin/activate
# 需已 prefetch 数据，或机器能访问 Hugging Face
bash scripts/run_official_toy_train_npu.sh
```
