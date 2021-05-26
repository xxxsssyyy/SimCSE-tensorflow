# SimCSE-tensorflow
SimCSE的tensorflow版本实现，以及基础实验对比

# Performances Comparison
Use `ATEC` dataset, the evaluation metric is `Spearman correlation`, use `[CLS]` of the last layer for a sentence sembeddings as the pooling strategy.
| models | dev | test |
| :------| :------ | :------ |
| Bert-base(Google) | 9.07 | 9.08 |
| SimCSE | 43.36 | 42.86 |
