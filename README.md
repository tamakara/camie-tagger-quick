# Camie-Tagger-Quick 🚀

针对 [Camais03/camie-tagger-v2](https://huggingface.co/Camais03/camie-tagger-v2) 模型的极简 Python 封装。

---

## 🛠️ 安装

```bash
git clone https://github.com/tamakara/camie-tagger-quick.git
cd camie-tagger-quick
pip install .
```

## 🚀 快速上手

```python
from PIL import Image
from camie_tagger_quick import CamieTagger

# 1. 初始化
tagger = CamieTagger(
    device="cpu", # 默认使用 CPU，如需 GPU 加速请设置 device="cuda"
    cache_dir="./model_cache", # 模型缓存目录，默认为系统全局缓存目录
    local_only=False # 是否仅使用本地缓存的模型文件，默认为 False
)

# 2. 执行
# threshold 参数用于调整阈值，默认为 0.61
# top_k 参数控制返回每个类别的最大标签数，默认为 50
results = tagger.tag(Image.open("image.png"), threshold=0.61, top_k=50)

# 3. 输出结果
# 使用内置的辅助函数进行格式化打印
tagger.print_results(results)

# 4. 访问标签数据
if 'character' in results:
    for item in results['character']:
        print(f"角色: {item['tag']}, 置信度: {item['confidence']:.2%}")
```

---

## ⚖️ 开源协议与鸣谢

- **模型来源**：权重由 [Camais03](https://huggingface.co/Camais03) 训练并发布。
- **本工具库**：基于 GNU GPL v3 协议开源。