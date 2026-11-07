# LG-DSD

**面向轻量级无人机遥感场景分类的 LLM 引导、混淆感知语义蒸馏**
（LLM-Guided Confusion-Aware Semantic Distillation for Lightweight UAV Remote Sensing Scene Classification）研究代码。

**作者：** Jianing Fang、Le Sun、Bandar Alghamdi 和 Mahmoud Ragab。

LG-DSD 利用学生模型的混淆模式，为轻量级遥感场景分类构造差异化语义监督。
仅使用训练集进行折外（out-of-fold，OOF）诊断，为每个类别选出一个有向易混淆类别。
由大语言模型（LLM）生成的描述用于区分目标类别及其选定的易混淆类别。
冻结的 RemoteCLIP 文本编码器将这些描述转换为语义原型。
学生模型通过分类损失、类别名称知识蒸馏和成对语义监督进行学习。
推理阶段仅保留学生模型的骨干网络和分类头。

## 目录

- [方法](#方法)
- [仓库结构](#仓库结构)
- [安装](#安装)
- [数据集划分](#数据集划分)
- [训练所需输入](#训练所需输入)
- [训练](#训练)
- [OOF 诊断](#oof-诊断)
- [发布内容范围](#发布内容范围)
- [引用](#引用)
- [许可证与致谢](#许可证与致谢)

## 方法

| 方法 | 监督方式 |
| --- | --- |
| KD / B0 | 分类和类别名称知识蒸馏 |
| B1 | KD 加通用语义监督 |
| B2 / LG-DSD | KD 加混淆条件下的差异化语义监督 |

B1 和 B2 使用相同的冻结 OOF 易混淆类别映射和 U1 成对目标函数。
二者的语义原型库不同：B1 使用通用描述，B2 使用差异化描述。
在 B2 中，负类使用易混淆类别自身的差异化原型；该原型是针对该类别所选易混淆类别构造的。
这种有向映射不一定是互相对应的。

训练目标为：

```text
L = (1 - alpha) * CE + alpha * T^2 * KL(teacher || student)
    + lambda_sem * L_sem
```

KD 中 `lambda_sem = 0`。
B1/B2 中 `lambda_sem = 0.03`、KD 温度 `T = 8`，语义温度为 `0.07`。
每个类别的原型由三个分别归一化的描述嵌入取平均得到，然后再对均值进行归一化。
仅用于训练的投影器依次执行 LayerNorm、1024 到 512 维的线性变换和 L2 归一化。
U1 使用二分类交叉熵对比 `bank[y]` 和 `bank[pi(y)]`，并将目标类别置于位置 0。

## 仓库结构

```text
.
├── README.md
├── requirements.txt
├── code/
│   ├── prepare_split_inputs.py
│   ├── train_unified_oof_semantic.py
│   ├── train_matched_kd.py
│   ├── AID_code/
│   │   ├── train_student.py
│   │   ├── generate_oof_confusions.py
│   │   └── prepare_aid_top1_transfer.py
│   ├── AIDER_code/scripts/
│   │   ├── build_aider_top1_oof.py
│   │   └── train_ce.py
│   └── RescueNet_code/scripts/
│       ├── build_rescuenet_oof_confusion.py
│       └── rescuenet_common.py
└── splits/
    ├── AID/
    │   ├── split.csv
    │   └── oof_folds.csv
    ├── AIDER/
    │   ├── train_val.csv
    │   ├── test.csv
    │   └── oof_folds.csv
    └── RescueNet/
        ├── train.csv
        ├── val.csv
        ├── test.csv
        └── oof_folds.csv
```

## 安装

请在独立环境中使用 Python 3.11 或更高版本。
在仓库根目录安装依赖：

```bash
python -m pip install -r requirements.txt
```

代码使用 PyTorch、torchvision、timm、NumPy、Pillow、scikit-learn 和 tqdm。
依赖列表列出代码所需的软件包；它不是历史训练环境的锁定文件。
论文方法使用 ImageNet 初始化的 MobileNetV3-Small 学生模型，以及冻结的 RemoteCLIP ViT-B/32 教师/文本编码器。

## 数据集划分

请从相应数据集提供方下载 AID、AIDER 和 RescueNet 图像。
仓库提供的 CSV 文件包含本研究实际使用的图像归属关系和 OOF 分配。
图像路径相对于对应的数据集根目录。

| 数据集 | 训练集 / 验证集 / 测试集 | 划分文件 | 仅训练集使用的 OOF 分配 |
| --- | --- | --- | --- |
| AID | 4502 / 498 / 5000 | [split.csv](splits/AID/split.csv) | [oof_folds.csv](splits/AID/oof_folds.csv) |
| AIDER | 4500 / 643 / 1286 | [train_val.csv](splits/AIDER/train_val.csv)、[test.csv](splits/AIDER/test.csv) | [oof_folds.csv](splits/AIDER/oof_folds.csv) |
| RescueNet | 3456 / 495 / 543 | [train.csv](splits/RescueNet/train.csv)、[val.csv](splits/RescueNet/val.csv)、[test.csv](splits/RescueNet/test.csv) | [oof_folds.csv](splits/RescueNet/oof_folds.csv) |

清单文件保留了图像归属、类别标签、分组标识和折编号。
对于 AIDER，固定开发集划分使用 `train_val.csv`，单独固定的测试集划分使用 `test.csv`。
复现 OOF 诊断时请保留原有分组分配。

### CSV 字段

| 字段 | 含义 |
| --- | --- |
| `relative_path` | 相对于下载后数据集根目录的图像路径 |
| `split` | 当多个划分共用一个文件时，标记为 `train`、`val` 或 `test` |
| `class_name`, `label` | AID/AIDER 的类别名称和从 0 开始的数值标签 |
| `label`, `label_id` | RescueNet 的类别名称和从 0 开始的数值标签 |
| `image_id` | RescueNet 图像的原始文件名 |
| `group_id` | 固定的分组标识 |
| `fold` | 从 1 开始编号的留出 OOF 折 |

RescueNet 的划分由文件名标识。
其类别顺序为 `superficial_damage`、`medium_damage` 和 `major_damage`。
OOF 文件仅包含训练图像。
对每个留出折，其余训练折构成该折辅助模型的拟合子集。

### 准备运行时图像路径

下载图像后运行路径辅助脚本：

```bash
python code/prepare_split_inputs.py --dataset AID --dataset-root /path/to/AID --output-dir inputs/AID
python code/prepare_split_inputs.py --dataset AIDER --dataset-root /path/to/AIDER --output-dir inputs/AIDER
python code/prepare_split_inputs.py --dataset RescueNet --dataset-root /path/to/RescueNet --output-dir inputs/RescueNet
```

对于 AID/AIDER，数据集根目录应包含各类别文件夹。
对于 RescueNet，根目录应包含 `raw/official/`。
该辅助脚本会添加运行时图像路径，同时保留原 CSV 中的所有值。
它不会读取图像，并且会拒绝覆盖已存在的运行时文件。
用户本地输入和生成的输出已通过 Git 忽略。

## 训练所需输入

运行学生模型训练前，请准备以下外部输入：

| 输入 | 所需格式 |
| --- | --- |
| 数据集图像 | 与所提供划分清单相对应的图像 |
| 类别名称教师缓存 | 包含 `teacher_logits`、`metadata` 和样本标识符 |
| 语义原型缓存 | `text_prototypes`，形状为 `C × 512`，并包含元数据 |
| 易混淆类别映射 | 每个类别对应一个非自身的目标类别/易混淆类别配对 |

AID/AIDER 的教师缓存样本标识符使用 `image_paths`，RescueNet 使用 `image_ids`。
请确保划分文件、教师 logits、原型张量和易混淆类别映射中的类别顺序一致。
易混淆类别条目需要包含 `target_label`、`opponent_label`，以及同时包含 `train_only` 和 `oof` 的 `source` 字段。

B1 使用通用语义库；B2 使用差异化语义库。
这两个方法应使用相同的教师缓存、数据划分、易混淆类别映射、随机种子和训练设置。
AIDER 使用论文中通过训练集拟合的温度加类别偏置教师校准方式。
RescueNet 使用论文中的未校准教师条件。
AID KD 使用其原始类别名称教师缓存，该缓存与 B2 的缓存不同。

## 训练

### 最终学生模型设置

| 设置 | 数值 |
| --- | --- |
| 学生模型 | ImageNet 初始化的 MobileNetV3-Small |
| 图像尺寸 | 224 × 224 |
| 训练轮数 / 批量大小 | 50 / 32 |
| 优化器 | AdamW |
| 学习率 / 权重衰减 | 0.0003 / 0.0001 |
| 学习率调度 | 余弦退火 |
| KD 温度 | 8 |
| KD 权重 `alpha` | AID：0.25；AIDER/RescueNet：0.10 |
| B1/B2 语义权重 / 温度 | 0.03 / 0.07 |
| 最终训练随机种子 | 42、43、44 |

训练时使用尺度为 0.7–1.0 的随机调整大小裁剪、水平翻转和 ImageNet 归一化。
评估时将图像缩放至 256，再进行 224 中心裁剪。
B1/B2 和对齐后的 KD 以验证集 Macro-F1 为首要指标、Accuracy 为次要指标选择模型；若指标完全相同，则保留较早的训练轮次。
AID KD 使用其原始的 Accuracy 优先选择策略；保留的检查点也会同时最大化记录的两项验证指标。
测试集仅用于最终评估。

### B2：LG-DSD

准备好外部输入后，可使用以下命令训练 AID 模型：

```bash
python code/train_unified_oof_semantic.py \
  --dataset AID --line U1 --seed 42 \
  --semantic-weight 0.03 --kd-alpha 0.25 \
  --split-file inputs/AID/split.csv \
  --class-teacher-cache inputs/AID/teacher.pt \
  --semantic-prototype-cache inputs/AID/B2_prototypes.pt \
  --confusion-pairs inputs/AID/opponents.json \
  --output-json outputs/AID/B2/summary.json \
  --output-predictions outputs/AID/B2/validation.npz \
  --checkpoint outputs/AID/B2/best.pt
```

对于 AIDER，设置 `--dataset AIDER`、`--kd-alpha 0.10` 和 `--split-file inputs/AIDER/train_val.csv`。
同时提供相应的教师缓存、原型、易混淆类别映射和输出路径。
对于 RescueNet，设置 `--dataset RescueNet` 和 `--kd-alpha 0.10`。
将 `--split-file` 替换为 `--train-csv`、`--val-csv` 和 `--dataset-root`，并指定对应输入。

论文中的 B1/B2 实验使用 U1。
请显式传入 `--semantic-weight 0.03`：原始 CLI 默认值为 `0.1`。
源代码中仍保留 U2/U3 研究分支。

### B1：通用语义对照

使用相同的 U1 命令，但改用通用原型缓存，并指定不同的输出路径。
保持类别名称教师、易混淆类别映射、数据划分和随机种子一致。
即使描述生成仅依赖类别名称，B1 训练仍然使用 OOF 易混淆类别映射。

### KD / B0

对于 AID，使用专用 KD 入口和原始类别名称教师缓存：

```bash
python code/AID_code/train_student.py \
  --mode kd --pretrained --seed 42 \
  --epochs 50 --batch-size 32 \
  --learning-rate 0.0003 --weight-decay 0.0001 \
  --kd-alpha 0.25 --temperature 8 --teacher-logit-scale 1 \
  --kd-gate none --kd-weighting none --kd-agreement-weighting none \
  --kd-warmup-epochs 0 --selection-metric accuracy --skip-test \
  --split-file inputs/AID/split.csv \
  --teacher-cache inputs/AID/kd_teacher.pt \
  --output-dir outputs/AID/KD
```

AIDER/RescueNet 使用 `code/train_matched_kd.py` 中可调用的 `train_matched_b0` 函数。
该方案保留统一的学生模型初始化，并使用 `0.9 * CE + 0.1 * KD`，不含语义损失。
以下示例训练所选数据集：

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("code").resolve()))
import train_matched_kd as kd

dataset = "AIDER"  # 若使用该数据集，请改为 "RescueNet"。
seed = 42
kd.HERE = Path("outputs/KD").resolve()
kd.ROOT = Path("/path/to/RescueNet").parent

method = {
    "dataset_details": {
        "AIDER": {
            "inputs": {
                "split_file": {"path": "inputs/AIDER/train_val.csv"},
                "class_teacher_cache": {"path": "inputs/AIDER/teacher.pt"},
            }
        },
        "RescueNet": {
            "inputs": {
                "train_csv": {"path": "inputs/RescueNet/train.csv"},
                "val_csv": {"path": "inputs/RescueNet/val.csv"},
                "class_teacher_cache": {"path": "inputs/RescueNet/teacher.pt"},
            }
        },
    }
}

kd.train_matched_b0(dataset, seed, method)
```

选择 RescueNet 时，请将数据路径设置为包含 `RescueNet/` 的父目录。
使用随机种子 43 和 44 重复最终训练，并为每次运行指定不同的输出目录。
AID KD 路径使用直接的 timm 分类头。
对齐后的 AIDER/RescueNet KD 以及 B1/B2 使用统一骨干网络和独立分类器。

## OOF 诊断

| 数据集 | 折数 | 辅助 CE 训练轮数 | 入口脚本 |
| --- | ---: | ---: | --- |
| AID | 5 折分层交叉验证 | 20 | [generate_oof_confusions.py](code/AID_code/generate_oof_confusions.py) |
| AIDER | 3 折分层、组间互斥 | 50 | [build_aider_top1_oof.py](code/AIDER_code/scripts/build_aider_top1_oof.py) |
| RescueNet | 3 折组间互斥，类别/样本均衡 | 50 | [build_rescuenet_oof_confusion.py](code/RescueNet_code/scripts/build_rescuenet_oof_confusion.py) |

OOF 随机种子 42 为最终训练随机种子 42–44 固定易混淆类别映射。
每张训练图像恰好获得一次留出预测。
选择非对角线计数最大的类别作为易混淆类别；若并列，则选择类别索引最小者。
AID 中无错误预测的行使用现有的 OOF 易混淆度排序。
[prepare_aid_top1_transfer.py](code/AID_code/prepare_aid_top1_transfer.py) 实现了 AID 的易混淆类别选择步骤。

启动辅助训练前，请先查看每个 OOF 入口脚本的 `--help`。
这些脚本会重新生成折分配；若要精确复现，请将生成结果与所提供的 `oof_folds.csv` 文件进行比较。
AID 的辅助脚本参数已在 CE 模式下显式提供，AIDER/RescueNet 的辅助模块也已包含在仓库中。

## 发布内容范围

本仓库提供方法代码、固定图像划分和原始 OOF 折分配。
数据集图像、生成的语义库、教师输出、原型张量和模型权重需由用户另行提供。
描述生成、完整缓存构建流程和统一的最终测试集评估脚本不属于本次发布内容。
仅凭当前发布的文件尚未验证完整端到端的重新训练。
实验日志、时间戳、内部审计材料和模型预测文件均未包含在仓库中。

## 引用

使用本代码时，请引用相关论文：

```bibtex
@unpublished{fang_lgdsd,
  author = {Fang, Jianing and Sun, Le and Alghamdi, Bandar and Ragab, Mahmoud},
  title = {{LLM-Guided Confusion-Aware Semantic Distillation for Lightweight UAV Remote Sensing Scene Classification}},
  note = {Manuscript}
}
```

## 许可证与致谢

本项目尚未指定软件许可证。
第三方软件、数据集和预训练模型仍受其各自的使用条款和署名要求约束。
本实现使用 PyTorch、torchvision、timm、NumPy、Pillow、scikit-learn、tqdm，以及基于 RemoteCLIP 的监督。
感谢 AID、AIDER 和 RescueNet 数据集的提供方。
