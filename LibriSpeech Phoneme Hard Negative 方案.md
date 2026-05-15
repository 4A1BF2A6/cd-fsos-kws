# LibriSpeech Phoneme Hard Negative 方案

## 目标

从 LibriSpeech alignment 的 `TextGrid` 中自动挖掘音素上接近 `Hey Camy` / `Hey Reco`、但语义上不应唤醒的短音频片段，作为 `_unknown_` hard negative 数据。

当前目标不是修改训练入口，而是先稳定产出一批可检查、可复现的数据集：

- wav 切片
- train/val/test manifest
- 类别统计和质量检查 report

## 脚本

新增脚本：

```text
tools/build_librispeech_phoneme_hardneg.py
```

脚本能力：

- 解析 Praat `TextGrid`
- 使用 `words` 和 `phones` 两层 alignment
- 跳过 `unaligned.txt` 中的 utterance
- 扫描 LibriSpeech `.flac` 并建立 `utt_id -> flac_path` 索引
- 按音素相似度挖掘 hard negative 候选
- 文本黑名单和完整 wake phrase 音素黑名单过滤
- 同 utterance 内 IoU 去重
- 按 category / speaker / phrase 控制数量
- 使用 `torchaudio` 切 wav，输出 16 kHz mono wav
- 生成 manifest 和 report
- 使用 `tqdm` 显示 `Mining TextGrid` 和 `Cutting audio` 进度条

如果环境没有 `tqdm`，脚本会自动退化为普通循环。

## 输入数据

alignment 根目录：

```text
/mnt/vdb1/logic/librispeech_alignments
```

LibriSpeech 音频根目录：

```text
/mnt/vdb1/logic/Librispeech/LibriSpeech
```

当前已验证 split：

```text
train-clean-100
```

## 输出数据

正式输出目录：

```text
/mnt/vdb1/logic/kws_hard_negative/librispeech_phoneme_hardneg_v1
```

输出结构：

```text
wavs/<category>/*.wav
manifests/all.csv
manifests/train.csv
manifests/val.csv
manifests/test.csv
reports/category_counts.csv
reports/blacklist_hits.csv
reports/missing_audio.csv
reports/parse_errors.csv
reports/duration_stats.csv
reports/sample_check_list.csv
```

manifest 字段：

```text
wav_path,label,category,target,source_dataset,source_split,source_utt_id,
speaker_id,chapter_id,start_sec,end_sec,duration_sec,words,phones,
normalized_phones,similarity,edit_distance,matched_rule,blacklist_flag
```

所有样本：

```text
label = _unknown_
```

## Hard Negative 类别

### hey_phoneme_similar

专门解决 `hey / hi` 类前缀误唤醒问题。

目标音素：

```text
HEY      HH EY
HI/HIGH  HH AY
HE       HH IY
```

典型样本：

```text
HAY   HH EY
HIGH  HH AY
HE    HH IY
HERE  HH IY R
```

这个类别是本轮新增的重点。模型如果听到 `hey`、`hi`、`high`、`he` 就容易误唤醒，应优先把这类样本作为 `_unknown_` hard negative 加入训练。

### camy_phoneme_similar

挖掘和 `Camy` 部分音素相似的词或连续 2 个词。

目标音素：

```text
K AE M IY
```

典型样本：

```text
CAMP
CAMEO
CATCH ME
CLAMMY
```

### reco_phoneme_similar

挖掘和 `Reco` 部分音素相似的词或连续 2 个词。

目标音素：

```text
R IY K OW
```

典型样本：

```text
REEK
REEKS
GREEKS
SHRIEKED
```

### local_phoneme_confuser

不按词边界挖掘，而是在 `phones` 层滑动 4-7 个音素窗口，找局部音素串接近完整 wake phrase 的片段。

它的作用是捕捉跨词、局部发音上像唤醒词的干扰。

典型样本：

```text
MAKE ME
TAKE ME
HE REQUIRED
```

### hey_nonwake

规则是 `HEY/HAY/HI/HIGH + 后续 1-2 个词`，再和完整 `Hey Camy` / `Hey Reco` 比较。

当前 `train-clean-100` 正式跑出的数量为 0。这不是脚本错误，而是 LibriSpeech 文本里满足该规则和音素阈值的样本很少。

后续如果需要增强这个类别，可以考虑：

- 放宽完整 wake phrase 相似阈值
- 接入其他语料
- 专门从包含 `hey / hi / high` 的文本语料中挖掘

## 正式生成结果

正式命令：

```bash
python tools/build_librispeech_phoneme_hardneg.py \
  --alignment_root /mnt/vdb1/logic/librispeech_alignments \
  --librispeech_root /mnt/vdb1/logic/Librispeech/LibriSpeech \
  --out_dir /mnt/vdb1/logic/kws_hard_negative/librispeech_phoneme_hardneg_v1 \
  --splits train-clean-100 \
  --sample_rate 16000 \
  --min_sec 0.5 \
  --max_sec 3.0 \
  --pre_pad_sec 0.10 \
  --post_pad_sec 0.15 \
  --similarity_threshold 0.55 \
  --max_edit_distance 2 \
  --max_per_category 1000 \
  --max_per_hey_phrase_category 100 \
  --num_workers 8 \
  --seed 1234
```

正式输出统计：

```text
Wrote 1496 final candidates

camy_phoneme_similar      308
hey_nonwake                0
hey_phoneme_similar       1000
local_phoneme_confuser    89
reco_phoneme_similar      99
```

这个分布符合当前目标：重点补强 `hey / hi` 前缀误唤醒。

如果要做更均衡的通用 hard negative 集合，可以重新生成时降低：

```text
--max_per_hey_phrase_category 30-50
```

或者训练时按 category 做采样平衡。

## 已完成验证

语法检查：

```bash
python -m py_compile tools/build_librispeech_phoneme_hardneg.py
```

dry-run 验证：

- 能解析 `train-clean-100` TextGrid
- 能索引对应 flac
- 能生成 manifest/report
- `missing_audio.csv` 无缺失
- `parse_errors.csv` 无解析错误

小规模真实切片验证：

- wav 成功写出
- 采样率为 16 kHz
- 时长在 `0.5s - 3.0s`

## 建议的训练接入方案

第一阶段建议不改模型结构，只改训练数据。

### 方案 A：直接混入 unknown

把本数据集 manifest 中的 wav 作为 `_unknown_` 样本加入训练。

优点：

- 实现最快
- 能直接验证 hard negative 是否降低误唤醒

风险：

- `hey_phoneme_similar` 数量占比高，可能影响普通 unknown 分布

建议：

- 先小规模实验
- 观察 `hey / hi / high / he` 误唤醒是否下降
- 同时观察真正 `Hey Camy / Hey Reco` 的 recall 是否下降

### 方案 B：按 category 平衡采样

训练时不要完全按文件数量采样，而是按 category 做均衡或加权。

建议初始比例：

```text
hey_phoneme_similar       40%
camy_phoneme_similar      25%
reco_phoneme_similar      20%
local_phoneme_confuser    15%
```

如果重点就是修复 `hey / hi` 误唤醒，可以提高：

```text
hey_phoneme_similar       50-60%
```

### 方案 C：两阶段实验

先做两个训练实验：

1. baseline：原训练数据
2. hardneg：原训练数据 + LibriSpeech phoneme hard negative

评估时重点看：

- `hey`
- `hi`
- `high`
- `he`
- `hay`
- `here`
- `camp`
- `cameo`
- `catch me`
- `reek / reeks / greeks`
- 正样本 `Hey Camy`
- 正样本 `Hey Reco`

判断标准：

- hard negative 误唤醒率下降
- 正样本 recall 不明显下降
- unknown / background 误拒不明显变差

## 质量检查命令

类别统计：

```bash
cat /mnt/vdb1/logic/kws_hard_negative/librispeech_phoneme_hardneg_v1/reports/category_counts.csv
```

时长统计：

```bash
cat /mnt/vdb1/logic/kws_hard_negative/librispeech_phoneme_hardneg_v1/reports/duration_stats.csv
```

查看每类 phrase 分布：

```bash
python - <<'PY'
import csv
from collections import Counter
from pathlib import Path

root = Path('/mnt/vdb1/logic/kws_hard_negative/librispeech_phoneme_hardneg_v1')
rows = list(csv.DictReader((root / 'manifests/all.csv').open()))
for cat in sorted(set(r['category'] for r in rows)):
    phrases = Counter(r['words'] for r in rows if r['category'] == cat)
    print('\n' + cat)
    for phrase, n in phrases.most_common(20):
        print(f'{n:4d} {phrase}')
PY
```

抽样试听列表：

```text
/mnt/vdb1/logic/kws_hard_negative/librispeech_phoneme_hardneg_v1/reports/sample_check_list.csv
```

## 当前结论

当前版本已经完成从 LibriSpeech alignment 到 hard negative wav/manifest/report 的闭环。

这一版最适合先用于修复 `hey / hi` 类前缀误唤醒，因为正式产出的 `hey_phoneme_similar` 样本最多，覆盖了 `HAY / HIGH / HE / HERE / HEY / HI` 等典型混淆。

下一步建议先接入训练做小规模对比实验，不急着继续扩脚本。训练结果如果显示 recall 下降，再回头调整 hard negative 采样比例或降低 `hey_phoneme_similar` 权重。
