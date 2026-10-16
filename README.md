# CLUE：弱监督增量分割实验代码

当前研究快照基于 SASA 精简基线，编码器为 DeiT-B/16。保留主辅分类损失 CLS、原始 PTC 和分割损失 SEG，在此基础上研究新旧类伪标签融合与蒸馏。

这份仓库是从4090上实际完成训练的代码和报告整理而来，不包含原始 A800 Git 历史。基础实现来源为 [SASA](https://github.com/ZhonggaiWang/SASA)，保留源文件中的作者信息。

## 当前方法

- **C**：局部背景恢复。
- **F**：用同一套分类特征比较新旧候选，cosine margin 为0.1。
- **普通 KD (`plain`)**：在有效像素上蒸馏旧前景类的 Bernoulli 分布，排除背景与新类通道。
- **策略一 (`counter`)**：保留普通 KD；对得到新类支持的像素，仅屏蔽旧模型正向支持的旧类通道，保留其他通道。
- **策略二 (`relations`)**：保留得到支持的高置信旧类正证据，以及高置信旧类负证据；按总通道权重调整蒸馏强度。
- 原证据像素筛选版本仍可通过 `--kd_mode evidence` 运行。

两个新 KD 策略不增加图片编码次数。原始 PTC、C 和 F 在本轮实验中保持不变。此前探索的其他模块仍保存在源码中，默认关闭，不属于下面的运行配置。

## 已完成结果

VOC 10-5，seed=0，每阶段8000步，2卡×每卡batch4，总batch8。**Step1和Step2的SEG都从第0步开始；C/F/KD从索引2000开始。** 最终阶段完整验证集为1449张。

| 方法 | Step1单尺度 | Step2单尺度 | Step2旧类含背景 | Step2新类 | Step2五尺度＋flip |
|---|---:|---:|---:|---:|---:|
| C＋F＋普通KD | 75.09 | **70.77** | **74.25** | **59.65** | **72.38** |
| C＋F＋策略一，完整重跑 | 74.94 | 70.65 | 74.19 | 59.35 | 72.24 |
| C＋F＋策略二 | 75.01 | 70.45 | 73.93 | 59.32 | 72.20 |

本轮两个新策略均未超过普通KD。所有结果只有一个训练seed，物理GPU对不同，不能把小幅差异理解为统计显著退化。策略一早先在Step2中途因共享内存错误退出的结果不作为最终结果；表中使用从Step1开始的完整重跑。

主要报告在 `kd2/reports/counter_restart_20261015/` 和 `kd2/reports/relations/`；配置、训练验证指标与KD累计统计在 `experiment_records/`。训练权重和原始图片、像素标注没有上传，`datasets/voc/` 中仅包含划分列表和图像级标签元数据。

## 环境

已验证环境：Python 3.9.19、PyTorch 2.3.1+cu118、torchvision 0.18.1+cu118、timm 0.5.4、NumPy 1.26.4、Pillow 10.2.0。

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

DenseCRF为可选项，需要另行安装 `pydensecrf`；普通训练和非CRF评估不需要它。

## Step0与数据

VOC数据根目录需要 `JPEGImages`，评估还需要 `SegmentationClassAug` 或 `SegmentationClass`。增量训练只使用RGB和当前新类的图像级标签，不读取增量图像的像素GT。

VOC10-5增量阶段统一使用**未来类别置背景0、Step0关闭PTC**的已确认权重：

```text
step0_bg0.pth
SHA256: 97ce07657378a28e6fa037576fcc57868992b19721e3be5d975b1858f9c94801
```

`STEP0_DEFAULT.json` 记录来源和校验和。权重需单独准备；不得使用不同任务划分的Step0。`--prev_checkpoint` 可显式传入本机权重路径，训练会验证其SHA256。其他任务需要先登记匹配的Step0，本快照不直接复用VOC10-5权重。

## 训练

先确认所选GPU空闲。以下脚本从指定Step0启动Step1，再以本次Step1最终权重启动Step2；不会重新训练Step0。

```bash
export PYTHON=python
export DATA_ROOT=/path/to/PascalVOC12
export STEP0_CHECKPOINT=/path/to/step0_bg0.pth
export CLUE_GPUS=4,5
export KD_MODE=plain          # plain / counter / relations / evidence
export RUN_NAME=my_plain_run  # 使用新名字，已有阶段目录不会覆盖
bash scripts/train_voc10_5.sh
```

两阶段各8000步，SEG始终启用，普通PTC权重0.2，SEG权重0.1，KD权重0.1、温度2。每2000步验证，但每阶段只保存最终权重。

源码位于 `kd2/`，输出位于 `kd2/runs/<RUN_NAME>/`。`run_kd2.py` 和 `restart_counter_full.py` 是原4090实验编排记录，包含该机器的路径与GPU安排；迁移到其他机器请使用上面的脚本。

## 评估

```bash
cd kd2
# 单尺度；也可使用最终训练日志中的完整验证结果
python eval_multiscale.py --checkpoint runs/my_plain_run/10-5/step2/model_final.pth \
  --data_folder /path/to/PascalVOC12 --scales 1 --output reports/my_plain_ss.json

# 五尺度＋水平翻转
python eval_multiscale.py --checkpoint runs/my_plain_run/10-5/step2/model_final.pth \
  --data_folder /path/to/PascalVOC12 --scales .5 .75 1 1.25 1.5 --flip \
  --output reports/my_plain_ms5_flip.json

# 可选DenseCRF；用 --help 查看组合参数
python eval_densecrf.py --help
```

评估输出文件不会覆盖。DenseCRF并不保证提升：已完成的普通KD实验中，五尺度flip不加CRF表现更好。

## 核验

```bash
cd kd2
python verify_kd.py
python verify_channel_kd.py
```

`source_manifest.json` 对应原训练源文件；`UPLOAD_SOURCE_MANIFEST.json` 记录这次上传时从服务器复制的文件哈希。新增README、依赖文件和便携训练脚本不在原训练源码清单中。
