# CompanyKWS 跨域少样本评估指南

本文档说明如何在公司自有唤醒词数据集（CompanyKWS）上，使用本仓库的预训练编码器做**跨域少样本（cross-domain few-shot）评估**。流程基于论文 *"Cross-Domain Few-Shot Open-Set Keyword Spotting Using Keyword Adaptation and Prototype Reprojection"* 实现。

---

## 1. 评估到底在做什么

整体推理流水线分三阶段：

```
预训练编码器 (MSWC 上训练)  →  目标域自适应 (CKA)  →  开/闭集分类 (Prototype Reprojection)
        ↑                              ↑                          ↑
   results/Pretrain_DSCNN_MSWC/    每个 episode 用 support      在 query 集上算指标
   best_model.pt                   集训练 adapter
```

- **预训练阶段（已完成）**：DSCNN 编码器在 MSWC 上以 episodic triplet loss 训过，权重存在 `results/Pretrain_DSCNN_MSWC/best_model.pt`。
- **目标域自适应（每 episode 重新做）**：冻结编码器，用 Custom Keyword Adapter (CKA) 做轻量自适应，仅用 support 集的 N-way × K-shot 几条样本。
- **少样本分类**：support 集出 prototype，query 集做最近原型分类；可选 `prototype_reprojection` 做开集校准。

每个 episode 随机抽 `n_way` 个唤醒词、每类 `n_support` 条做支持集；其余作为 query 集。重复 `n_episodes` 次取平均。

---

## 2. 数据集要求

期望的目录布局：

```
<root>/
└── <wake>/                                  # 唤醒词目录 = 类别
      ├── <emp_id>/                          # 员工 / 说话人 ID
      │     └── <env>_<dist>_<speed>_<take>/ # 一次采集 take 内多通道同步
      │           ├── ch01.wav … ch04.wav    # 阵列原始通道
      │           ├── ch05.wav               # 级联信号
      │           ├── ch06.wav               # 回采信号
      │           └── ch07.wav               # DSP 增强后信号（默认推理通道）
      └── <env>_background/                  # 背景噪声（不属于具体说话人）
            └── *.wav
```

**约束：**

- 每个 wake 目录至少要有 ≥ 3 个 `<emp_id>` 子目录，否则 speaker-disjoint 切分会退化（默认 80/10/10 比例）。
- 单个 wav 默认按 16 kHz 单声道处理；若原始采样率不同，wrapper 会自动 `torchaudio.functional.resample`。
- `<env>_background/` 默认作训练/评估时的噪声混入（对齐 GSC 的 `_background_noise_` 语义）；当传入 `--speech.include_unknown` 时，wrapper 会把背景切成定长片段额外作为 `_unknown_` 类参与开集评估。
- `<wake>` 目录名直接当作类别名使用，不区分大小写要看你自己的命名。

---

## 3. 评估模式一：直接推理评估（最快路径）

直接用 MSWC 预训练 checkpoint，**不**额外训练，仅做 few-shot 评估。适合验证「现有编码器能否迁移到公司数据」。

### 3.1 命令模板

```bash
python target_adapting_querying.py \
    --data.cuda \
    --choose_cuda 0 \
    --model.model_path results/Pretrain_DSCNN_MSWC/best_model.pt \
    --speech.dataset CompanyKWS \
    --speech.task CompanyKWS_ALL \
    --speech.default_datadir /path/to/wakeword_dataset/ \
    --speech.channel ch07 \
    --fsl.test.batch_size 64 \
    --fsl.test.n_support 5 \
    --fsl.test.n_way 2 \
    --fsl.test.n_episodes 50 \
    --querying.prototype_reprojection
```

### 3.2 关键参数

| 参数 | 含义 | 调参建议 |
|---|---|---|
| `--speech.dataset` | 数据集类型 | 固定 `CompanyKWS` |
| `--speech.task` | 任务名 | `CompanyKWS_ALL` 用全部 wake；或 `wake_a,wake_b,...` 显式列举 |
| `--speech.default_datadir` | 数据根目录 | 必须以 `/` 结尾，指向上节布局的 `<root>/` |
| `--speech.channel` | 取哪一路通道 | 默认 `ch07`（DSP 增强）；想看原始阵列效果改 `ch01` |
| `--speech.crop_strategy` | 变长音频裁剪策略 | `center` 取几何中段；`energy` 用 1s 滑窗找 RMS 能量峰值（默认 `center`，唤醒词尾音重要时建议 `energy`） |
| `--speech.merge_val` | 验证集去向 | `none` 不动；`train` 合进训练集；`test` 合进测试集（默认 `none`，仓库本来不用 val） |
| `--speech.include_unknown` | 是否启用 `_unknown_` 类 | 设了之后从背景切片生成 `_unknown_` 样本，触发开集评估 |
| `--fsl.test.n_way` | 每个 episode 几路分类 | **不能超过实际 wake 数 (+1 if include_unknown)**；纯闭集 task 设成等于 wake 数即可 |
| `--fsl.test.n_support` | 每类支持样本数 | 5/10 是常用值；少样本场景就该 ≤ 10 |
| `--fsl.test.n_episodes` | 评估 episode 数 | 50–100 之间，越多波动越小 |
| `--fsl.test.batch_size` | query 推理 batch | 看显存，默认 64–128 |
| `--querying.prototype_reprojection` | 开启原型重投影 | 闭集也开着，对结果无伤；开集场景必开 |

### 3.3 切换通道做对比

公司硬件给了 7 路通道：`ch01–ch04` 阵列原始、`ch05` 级联、`ch06` 回采、`ch07` DSP 增强。同一条命令换 `--speech.channel` 即可对比通道间差异：

```bash
# baseline: DSP 后
--speech.channel ch07

# 看原始通道之一
--speech.channel ch01
```

建议每跑一次把 `--log.exp_dir` 也跟着改，避免互相覆盖结果。

---

## 4. 评估模式二：在公司数据上重新预训练后再评估

适合「MSWC 预训练域差太大，想在公司数据上从头跑源域预训练」。两步：

### 4.1 步骤 A：源域预训练

```bash
python source_pretraining.py \
    --data.cuda \
    --speech.dataset CompanyKWS \
    --speech.task CompanyKWS_ALL \
    --speech.default_datadir /path/to/wakeword_dataset/ \
    --speech.channel ch07 \
    --train.epochs 40 \
    --train.n_way 2 \
    --train.n_query 10 \
    --train.n_episodes 200 \
    --log.exp_dir results/Pretrain_DSCNN_CompanyKWS
```

注意：

- `--train.n_way` 不能超过实际 wake 数。如果只有 2 个 wake，episodic triplet 的难度会很低，正负样本对几乎平凡——这种情况下**模式一更划算**。
- 训练完 `best_model.pt` 会写到 `results/Pretrain_DSCNN_CompanyKWS/`。

### 4.2 步骤 B：用新 checkpoint 跑评估

把模式一命令里的 `--model.model_path` 换成新的：

```bash
--model.model_path results/Pretrain_DSCNN_CompanyKWS/best_model.pt
```

其它参数同模式一。

---

## 5. 数据切分逻辑（speaker-disjoint）

`data/CompanyKWS.py` 里：

- 收集所有 `<emp_id>`，固定随机种子 shuffle 后按 `validation_percentage` / `testing_percentage`（默认 10% / 10%）切成 train / val / test 三组。
- **同一 emp_id 的所有 take 永远在同一个 split 里**，避免说话人泄漏。
- 当 `<emp_id>` 总数 < 3 时会退化：测试集 = 全部 ÷ 3 向上取整、训练集至少 1 个，val 可能为空。这种时候少样本指标的统计意义有限，仅作冒烟测试用。
- 仓库现有评估流程不真正消费 val（support 从 training 抽，query 走 testing）。如果你想让 val 不浪费：用 `--speech.merge_val test` 把 val 合并进 test 扩大评估面，或 `--speech.merge_val train` 把它喂给 CKA 自适应。

启动日志会打印实际切分：

```
[CompanyKWS] channel=ch07 | wakes=2 | speakers train/val/test = 5/1/1 | samples train/val/test = 419/84/48
```

如果 testing 那栏 = 0，把数据再加点，或调 `validation_percentage` / `testing_percentage`（目前在 `data/CompanyKWS.py` 里硬编码，可改成从 args 读）。

---

## 6. 输出与指标解读

评估结束后，结果会写到 `--model.model_path` **同级目录**下的 trace / log 文件。打印的关键指标：

- `accuracy_pos`：正类（真实 wake）闭集准确率，不考虑开集拒识。
- `accuracy_neg`：负类（`_unknown_`）被正确判成 unknown 的比例，仅当传了 `--speech.include_unknown` 才有意义。
- `aucROC`：ID/OOD 区分能力——衡量正样本分数排在负样本之前的概率，与阈值无关。仅当 `accuracy_neg` 有效时才有意义。
- `acc_far05`：把判定阈值卡到「FAR ≤ 5%」时的正类准确率（FAR = 负样本被误唤醒的比例）。
- `frr_far05`：同一工作点下的 FRR（真实唤醒被漏识的比例）。
- `thr_far05`：满足 FAR ≤ 5% 的判定阈值。
- `cerr_far05`：该工作点下的综合错误率（保留字段，当前实现里恒为 0）。

> 命名说明：之前版本里这几个键叫 `*_prec95`，但这个工作点对应的是 specificity=95% / FAR=5%，跟 precision 没关系，已重命名为 `*_far05`。

只有 closed-set 一栏有值（即 `accuracy_neg = 0`、`aucROC = 0`）时，说明你这次跑的是纯闭集 task —— 不传 `--speech.include_unknown` 就是这种。

---

## 7. 常见错误排查

### 7.1 `TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'`

脚本里某处假设一定有 `_unknown_` 类。当前仓库已修两处（`target_adapting_querying.py` 第 303 行的支持集切片、第 75 行的 OOD 概率累加）。如果你拉了更早版本再次踩到，搜 `unk_idx` 加 None 判空即可。

### 7.2 `Background sample is too short!` / `No wav files found for channel chXX`

- 通道名拼错：检查 `--speech.channel` 是否真的是 `chNN.wav` 形式。
- 通道文件确实不存在：`ls <root>/<wake>/<emp_id>/<take>/` 看一下。
- 背景太短（< 1 秒）：wrapper 已做兜底，单条短背景会被跳过，不会再抛异常。

### 7.3 `n_way > num_classes`

任何 episode 的 `n_way` 都不能大于 wake 数 + (1 if include_unknown else 0)。把 `--fsl.test.n_way` 调下来。

### 7.4 GPU 上 `mfcc` 设备不一致

代码已自动处理：在 `model.cuda()` 后会再单独 `model.preprocessing.mfcc.cuda()`（MFCC 模块不会随 `model.cuda()` 一起搬）。如果你魔改过 model wrapper，记得保留这一行。

### 7.5 切分里 testing 为 0 / 数据偏少

参考 §5。临时方案：把 `data/CompanyKWS.py` 里 `params['testing_percentage']` 调到 20% 或 30%；或加 `--speech.merge_val test` 把 val 合进来。长期方案：补数据。

### 7.6 `KeyError: 'bg_offset_seconds'` (集合器报错)

跨类 batch 里部分 record 来自 `_unknown_`（背景切片，带 `bg_offset_seconds`），部分来自 wake（不带），torchnet 默认 collator 要求字段一致。已在 `CompanyKWS.load_audio` 里 `dict(d).pop('bg_offset_seconds', None)` 兜底——load 完所有 record schema 一致即可。如果改了加载逻辑导致复现，记得保留这个 pop。

### 7.7 `conv_cka object has no attribute 'delta'`

当 `--adapting.cka-opt` 里没有 `delta`（例如 `beta`-only）时，`conv_cka.__init__` 会跳过创建 `self.delta`，但 `forward` 里仍引用。已在 `models/CKAs_module.py` 的 `forward` 入口加 `if self.ad_type == 'none': return y` 提前返回。

### 7.8 `LayerNorm` shape mismatch（修改 `--speech.clip_duration` 后）

`results/Pretrain_DSCNN_MSWC/best_model.pt` 里 LayerNorm 的 `normalized_shape=[276, 13, 5]` 是 1 秒输入算出来的；改 `--speech.clip_duration 2000` 之类会让前向时 shape 不一致。**别改时长**——改用 `--speech.crop_strategy energy` 让 1 秒窗口落在能量峰值上即可。

---

## 8. 该项目本次新增/修改文件速查

| 文件 | 作用 |
|---|---|
| `data/CompanyKWS.py` | 新数据集 wrapper（变长裁剪、speaker-disjoint 切分、可选 `_unknown_` 背景切片） |
| `parser_kws.py` | 新增 `--speech.channel` / `--speech.crop_strategy` / `--speech.merge_val` |
| `target_adapting_querying.py` | 新增 `CompanyKWS` 分发分支 + 修闭集兼容性 bug + 负集 loader fallback |
| `source_pretraining.py` | 放开 `speech.dataset` 写死覆盖 + 新增 `CompanyKWS` 分发分支 |
| `metrics.py` | 闭集（`y_score_neg=None`）兜底、`*_prec95` → `*_far05` 重命名 |
| `models/CKAs_module.py` | `ad_type='none'` 时 `forward` 直接返回，避免引用未创建的 `self.delta` |
| `CompanyKWS_EVAL.md` | 本文档 |

---

## 9. 后续可调方向

- **多通道融合**：当前只取一路。如需 beamforming / 多通道拼输入，改 `data/CompanyKWS.py` 的 `_list_take_wavs` 与 `load_audio`，把 `desired_samples` 维度从 `(1, T)` 改成 `(C, T)`，并相应调编码器输入维度。
- **更难的开集负集**：当前 `_unknown_` 来自背景切片（噪声 vs 语音的判别太容易，AUROC 容易拉满）。可以接 GSC 的非唤醒词（yes/no/stop 等）做"人声非唤醒词"挑战集，让 FAR 维度真正区分模型。
- **任务粒度**：当前 `--speech.task` 只支持 `CompanyKWS_ALL` 或显式列举。若要做 GSC12 / GSC22 风格的 known/unknown 划分，在 wrapper 顶部加一张 `TASKS` 字典即可。
- **RTF / FLOPs 统计**：本仓库已有 `demo_fewshot_wav.py`，可类比改造一个 `demo_fewshot_companykws.py` 测推理时延。
