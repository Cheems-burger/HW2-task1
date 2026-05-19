# 基于预训练 ResNet18 微调及注意力机制的花朵分类

本项目在 Oxford-102 Flowers 数据集上，以 ImageNet 预训练的 ResNet18 为基线，系统研究了迁移学习、超参数优化、消融实验以及 CBAM 注意力机制对花朵分类性能的影响。最终最佳模型（ResNet18 + CBAM）在测试集上达到 **90.76%** 的准确率。

## 仓库结构

HW2-task1/
├── README.md # 项目说明
├── requirements.txt # 依赖列表
├── baseline.py # 任务一：Baseline（预训练微调）
├── grid_search.py # 任务二：网格搜索（超参数分析）
├── ablation_scratch.py # 任务三：消融实验（随机初始化）
└── task4_cbam.py # 任务四：CBAM 注意力机制

## 环境配置

### 安装依赖

```bash
pip install -r requirements.txt

torch>=2.0.0

torchvision>=0.15.0

scipy

matplotlib

swanlab
# 代码说明
baseline.py – 预训练微调基线
grid_search.py – 6组超参数网格搜索
ablation_scratch.py – 随机初始化消融实验
cbam.py – 集成CBAM注意力机制
