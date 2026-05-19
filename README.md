# 基于预训练 ResNet18 微调及注意力机制的花朵分类
本项目在 Oxford-102 Flowers 数据集上，以 ImageNet 预训练的 ResNet18 为基线，系统研究了迁移学习、超参数优化、消融实验以及 CBAM 注意力机制对花朵分类性能的影响。
# 环境依赖
- torch>=2.0.0
- torchvision>=0.15.0
- scipy
- matplotlib
- swanlab
# 代码说明
- baseline.py – 预训练微调基线
- grid_search.py – 6组超参数网格搜索
- ablation_scratch.py – 随机初始化消融实验
- cbam.py – 集成CBAM注意力机制

## 仓库结构

```text
.
├── README.md
├── baseline.py
├── grid_search.py
├── ablation_scratch.py
└── cbam.py
