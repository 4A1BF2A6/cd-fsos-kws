# Few-shot KWS 演示

这个 demo 使用仓库里已经提供的 MSWC 预训练 backbone：
`results/Pretrain_DSCNN_MSWC/best_model.pt`。

它不要求公司的正式唤醒词数据已经录制完成。你可以先临时录几条 wav，
按类别放到文件夹里，用来演示“少量样本注册唤醒词，然后识别 query 音频”的流程。

## 目录结构

```text
demo_data/
  support/
    wakeword/
      wake_001.wav
      wake_002.wav
      wake_003.wav
    background/
      bg_001.wav
      bg_002.wav
      bg_003.wav
  query/
    test_001.wav
    test_002.wav
```

演示唤醒词时，可以这样准备：

- `wakeword/`：放 3-10 条临时录制的唤醒词音频。
- `background/`：放普通语音、静音、噪声，或者和唤醒词相似但不应该唤醒的音频。
- `query/`：放待测试的音频，可以包含正样本和负样本。

音频建议使用 16 kHz、单声道 wav。脚本会自动重采样，并把音频裁剪或补零到 1 秒。

## 运行方式

```bash
python demo_fewshot_wav.py \
  --support_dir demo_data/support \
  --query demo_data/query \
  --model_path results/Pretrain_DSCNN_MSWC/best_model.pt \
  --threshold 0.65
```

Windows PowerShell 可以写成：

```powershell
python demo_fewshot_wav.py `
  --support_dir demo_data/support `
  --query demo_data/query `
  --model_path results/Pretrain_DSCNN_MSWC/best_model.pt `
  --threshold 0.65
```

输出会展示：

- 当前注册了哪些类别。
- 每条 query 音频被判定为哪个类别。
- 类别分数。
- query 到各个类别 prototype 的距离。

`--threshold` 用来做拒识。如果最高分低于阈值，脚本会输出 `_reject_`。
这个阈值需要结合你自己的负样本调试，演示阶段可以先从 `0.65` 开始试。

## 推荐演示流程

1. 先录 3-5 条唤醒词，放到 `demo_data/support/wakeword/`。
2. 再录 3-5 条普通话术、相似词或静音，放到 `demo_data/support/background/`。
3. 把待测试音频放到 `demo_data/query/`。
4. 运行脚本，观察输出结果。
5. 调整 `--threshold`，展示误唤醒如何被拒识。

## 演示时可以强调的点

- 不需要完整训练集，也能快速注册一个新唤醒词。
- 每个类别只需要少量 support 样本，就能形成 prototype。
- query 音频会和各类别 prototype 做距离比较。
- 阈值可以控制“唤醒”和“拒识”的平衡。
- 公司正式录音完成后，可以把临时 wav 替换成真实数据继续迭代。

## 注意事项

- 这个 demo 是为了快速展示流程，不等同于最终商用唤醒系统。
- 正式上线前需要准备更多正样本、近似负样本和长音频背景测试集。
- 如果当前环境没有安装 `torch` / `torchaudio`，需要先按 README 配好项目环境。

## 公司正式唤醒词数据到位后怎么办

公司录音数据交付后，不建议直接全部塞进 `support/` 里跑 demo。
更稳妥的做法是先整理数据、划分集合、评估阈值，再决定是否进入正式训练。

### 1. 先检查数据

拿到音频后，先确认这些信息：

- 音频格式：建议统一为 wav、16 kHz、单声道。
- 每条音频是否只包含一个唤醒词。
- 是否有过长静音、截断、爆音、明显录错词。
- 文件名或标注里是否能区分说话人。
- 是否包含录音设备、距离、环境噪声等信息。

如果原始音频不是 16 kHz 单声道，可以先保留原始文件，再额外导出一份模型使用的标准 wav。

推荐整理成：

```text
company_data_raw/
  wakeword/
  hard_negative/
  background/

company_data_16k/
  wakeword/
  hard_negative/
  background/
```

其中：

- `wakeword/`：真实唤醒词正样本。
- `hard_negative/`：相似词、错读词、只包含部分音节的音频。
- `background/`：普通业务语音、环境音、静音、非唤醒词语音。

### 2. 按说话人划分数据集

不要按音频随机划分。应该按说话人划分，避免同一个人的声音同时出现在训练和测试里。

推荐：

```text
train speakers      70%
validation speakers 15%
test speakers       15%
```

可以整理成：

```text
company_data_split/
  train/
    wakeword/
    hard_negative/
    background/
  val/
    wakeword/
    hard_negative/
    background/
  test/
    wakeword/
    hard_negative/
    background/
```

如果暂时说话人数很少，至少保留一批从未参与调参的测试说话人。

### 3. 先用当前 demo 做快速验证

可以先从 `train/wakeword/` 里挑 5-20 条作为 support，
从 `train/background/` 和 `train/hard_negative/` 里挑一些作为 background support。

示例：

```text
demo_data/
  support/
    wakeword/
    background/
  query/
```

然后把 `val/` 或 `test/` 里的音频放进 `query/` 跑：

```powershell
python demo_fewshot_wav.py `
  --support_dir demo_data/support `
  --query demo_data/query `
  --model_path results/Pretrain_DSCNN_MSWC/best_model.pt `
  --threshold 0.65
```

这个阶段主要看：

- 唤醒词正样本能不能稳定识别。
- hard negative 是否容易误触发。
- 阈值大概应该设置到多少。
- 哪些说话人、设备、环境最容易失败。

### 4. 做正式评估

正式评估不要只看单条输出，建议统计这些指标：

- FRR：说了唤醒词但没有唤醒的比例。
- FAR：没说唤醒词但误唤醒的比例。
- hard negative 误触发率。
- 不同设备、距离、噪声环境下的分组表现。
- 长音频上的误唤醒次数，例如每小时误唤醒多少次。

如果只是内部 demo，可以先用准确率和误触发案例分析。
如果准备上线，就必须做长音频误唤醒测试。

### 5. 决定后续路线

数据量较少时，可以继续用当前 few-shot prototype 方案：

- 优点：开发快，新唤醒词接入快。
- 缺点：误唤醒控制能力有限，需要依赖阈值和负样本质量。

数据量足够后，更建议训练一个专用 KWS 模型：

- 输入：1 秒左右的音频窗口。
- 类别：`wakeword`、`unknown/background`，必要时加 `hard_negative`。
- 模型：可以继续用 DSCNN，也可以换成 TC-ResNet、CRNN 等轻量模型。
- 输出：唤醒分数，再配合阈值、平滑和 VAD 做上线逻辑。

实际项目可以分两步走：

1. 先用 `demo_fewshot_wav.py` 快速验证唤醒词可行性。
2. 数据积累到一定规模后，再训练专用 KWS 模型并做端侧/服务端部署评估。

### 6. 数据交付给开发时的建议格式

建议公司录音团队交付时包含一个 metadata 表：

```csv
file,speaker_id,text,label,device,distance,noise_type,split
wake_0001.wav,S0001,你好小智,wakeword,phone,near,quiet,train
wake_0002.wav,S0002,你好小智,wakeword,phone,far,office,val
neg_0001.wav,S0003,你好小志,hard_negative,phone,near,quiet,test
bg_0001.wav,S0004,今天天气不错,background,phone,near,office,test
```

有了这个表，后续就可以更容易地做数据清洗、分组评估和训练集构建。
