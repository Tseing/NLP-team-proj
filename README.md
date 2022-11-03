# 使用方法
1. 创建虚拟环境，进入虚拟环境后，使用 `pip install -r requirements.txt` 安装依赖；
2. 在根目录创建 `dataset` 文件夹，放入 `Train.csv` 文件；
3. 在虚拟环境中使用 `python train_better_rnnlm.py` 开始训练。

- 若使要使用 GPU 训练，需要安装 GPU 对应版本的 CUDA 与 Cupy，并把 `train_better_rnnlm.py` 中的注释去除。