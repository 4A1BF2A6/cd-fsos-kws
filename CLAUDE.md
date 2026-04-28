# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

Adapt-KWS：论文 "Cross-Domain Few-Shot Open-Set Keyword Spotting Using Keyword Adaptation and Prototype Reprojection"（ICASSP 2025）的官方 PyTorch 实现。整体流程分两步：先在 MSWC 上对 DSCNN 特征编码器做源域预训练，再针对每个目标域用 Custom Keyword Adapters (CKAs) 做少样本自适应，最后用 prototype reprojection 做开集分类。

## 运行环境

依赖 `python=3.7.12`、`torch==1.13.1`、`torchaudio==0.13.1`，安装方式 `pip install -r requirements.txt`。注意 `requirements.txt` 是冻结的环境快照，里面有大量与本项目无关的包（Django、transformers、deepspeed 等），不要因为某个包出现在文件里就以为它被实际使用。

## 常用命令

源域预训练（MSWC，DSCNN 编码器，online triplet loss）：
```
python source_pretraining.py --data.cuda \
  --speech.default_datadir <dataset_path>/MSWC/en/ \
  --train.epochs 40 --train.n_way 80 --train.n_query 20 --train.n_episodes 400 \
  --log.exp_dir <output_dir>/<EXP_NAME>
```

目标域自适应 + 开集查询（以 GSC 为例）：
```
python target_adapting_querying.py --data.cuda --choose_cuda 0 \
  --model.model_path results/Pretrain_DSCNN_MSWC/best_model.pt \
  --speech.dataset googlespeechcommand --speech.task GSC12,GSC22 \
  --speech.default_datadir <dataset_path>/GSC/speech_commands_v0.02/ \
  --speech.include_unknown \
  --fsl.test.batch_size 264 --fsl.test.n_support 10 --fsl.test.n_way 11 \
  --fsl.test.n_episodes 100 \
  --querying.prototype_reprojection
```
`speech.dataset` 取值 {`googlespeechcommand`, `UASpeech`, `MDSC`}，需要和对应的 `speech.task`（`GSC12,GSC22` / `UASpeech12,UASpeech22` / `MDSC12,MDSC22`）以及 `speech.default_datadir` 配套使用。测试结果默认写到 `model.model_path` 同级目录。

仓库里**没有**自动化测试、lint 配置或构建步骤——入口就是上面两个脚本（`train_class_loss.py` 是较少使用的第三个训练变体）。

## 参数约定

所有命令行参数定义在 `parser_kws.py`，统一使用**点号分组命名**（如 `--model.model_name`、`--train.n_way`、`--fsl.test.n_support`、`--adapting.cka-opt`）。`utils.filter_opt(opt, 'speech')` 会把前缀剥掉，得到该子系统的参数字典。新增参数请沿用 `<group>.<name>` 风格；如果新增了分组，记得同步检查 `filter_opt` 的调用点。

两个训练脚本在 `parse_args` 之后会**强制覆盖**部分参数（例如 `source_pretraining.py` 会把 `speech.dataset` 改为 `'MSWC'`，`model.model_name` 改为 `'repr_conv'`，`model.encoding` 改为 `'DSCNNL_LAYERNORM'`，`train.loss` 改为 `'triplet'`）。光改 CLI 默认值不够，必须看脚本主体里的覆盖逻辑。

## 代码结构

- `parser_kws.py`——所有 CLI 选项的唯一来源。
- `source_pretraining.py`——episodic 预训练主循环。通过 `models.utils.get_model` 构建 `repr_conv` 模型，每个 episode 调用 `model.loss(samples)`，每 10 个 episode 存一次 checkpoint，每个 epoch 把 `best_model.pt` 写到 `--log.exp_dir`。
- `target_adapting_querying.py`——加载预训练得到的 `ReprModel`，用 `models.CKAs_module.ReprModel_cka` 包一层，使得编码器冻结、CKA 参数可训练，可训练子集由 `models.adapting_para_selection.get_cka_params` 决定；逐 episode 做自适应后，用 `classifiers/` 里的某个分类器（`NCM` / `NCM_openmax` / `peeler` / `dproto` / `prototype_reprojection`）评估。`metrics.compute_metrics` 输出闭集/开集指标。
- `models/`——模型构建。`MODEL_REGISTRY` 通过 `models/e2e.py` 和 `models/repr_model.py` 里的 `@register_model` 装饰器填充，而这两个模块由 `models/__init__.py` 顺带 import，注册表才不会是空的。`get_model(model_opt)` 按 `model_name` 分发。
  - `models/encoder/`——backbone（`DSCNN*`、`Res8/15`、`TCResNet8`）。`DSCNNL_LAYERNORM`、`DSCNNS_PEELER` 等变体对应特定 recipe。
  - `models/losses/`——`triplet`、`protonet`、`angproto`、`amsoftmax`、`peeler`、`dproto` 等损失，由 `ReprModel.__init__` 中的 `criterion['type']` 选择。
  - `models/preprocessing.py`——MFCC 前端。当 `data.cuda` 为 True 时，`model.preprocessing.mfcc` 需要单独 `.cuda()`，不会随 `model.cuda()` 一起搬。
  - `models/CKAs_module.py` + `models/adapting_para_selection.py`——Custom Keyword Adapter 包装层。`--adapting.cka-ad-type`、`--adapting.cka-ad-form`、`--adapting.cka-opt`、`--adapting.cka-init` 控制 adapter 的形状、初始化方式以及哪些参数参与训练。
- `classifiers/`——可替换的少样本分类器（`NearestClassMean`、`NCMOpenMax`、`PeelerClass`、`DProto`、`prototype_reprojection`）。统一接口为 `evaluate_batch(x, labels, return_probas=...)`，并都暴露 `word_to_index`；`_unknown_` 是开集 / OOD 类。
- `data/`——每个数据集一个 wrapper：`MSWC.py`（源域），`GSC.py`、`UASpeech.py`、`MDSC.py`（目标域），加上 `data_utils.py`。每个 wrapper 暴露 episodic dataloader（如 `ds.get_episodic_dataloader(split, n_way, n_support+n_query, n_episodes)`），返回 `{'data', 'label'}` 批次。**注意**：`source_pretraining.py` 里 import 的是 `data.GSCSpeechData.GSCSpeechDataset` 和 `data.MSWC.MSWCDataset`，如果某个 import 路径和实际文件名对不上，那是仓库本身的遗留问题，不要随手重命名。
- `results/Pretrain_DSCNN_MSWC/`——已附带的预训练 checkpoint（`best_model.pt`）+ `opt.json`（训练起始时保存的完整参数快照）+ `trace.txt`（`log.py` 写的逐 epoch JSON-lines 日志）。

## 数据目录布局

两个训练脚本期望的磁盘结构在 `README.md` 的 "Data Preparation" 部分有说明：`<dataset_path>/MSWC/en/`（含 `en_{train,test,dev}.csv` 与 `clips_wav/`）、`<dataset_path>/MSWC/noise/`（用于混入 DEMAND 噪声）、目标数据集分别放在 `<dataset_path>/GSC/`、`<dataset_path>/UASpeech/`、`<dataset_path>/MDSC/`。默认假设 MSWC 已经把 opus 转成 wav；如果跳过这一步，需要把 `data/MSWC.py` 大约第 390 行的文件夹名改回 `clips`。

## CUDA / 环境怪癖

两个脚本在 import 阶段会执行 `os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('_CONDOR_AssignedGPUs', 'CUDAx').replace('CUDA','')`，这是 HTCondor 集群留下的写法。`target_adapting_querying.py` 之后会再用 `--choose_cuda` 覆盖一次。在普通工作站上这行其实没有副作用，只是看着别扭——不要随便"清理"掉，先确认下游没有依赖该环境变量在 import 早期就被设置。
