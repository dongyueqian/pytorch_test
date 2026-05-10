### 环境版本
- Python：3.11.9 
- torch：2.2.2（Intel Mac 最高稳定版） 
- transformers：4.35.2 
- sentence-transformers：2.2.2 
- numpy：<2 旧版兼容 
- huggingface_hub：0.17.1

### 可用方案亮点
- 不用自己手动下残缺模型、不用管相对 / 绝对路径； 
- 用 uer/sbert-base-chinese-nli 中文专用模型，语义相似度很准； 
- 第一次自动下载到本地缓存，后续完全离线运行； 
- 所有警告、找不到模型提示全部屏蔽，输出干净； 
- 完美适配你 Intel Mac，后续安心学 PyTorch、做语义匹配测试都没问题。