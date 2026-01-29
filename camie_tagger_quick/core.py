import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import onnxruntime as ort
import torchvision.transforms as transforms
from PIL import Image
from huggingface_hub import hf_hub_download

# 常量配置
MODEL_REPO = "Camais03/camie-tagger-v2"
MODEL_FILE_NAME = "camie-tagger-v2.onnx"
METADATA_FILE_NAME = "camie-tagger-v2-metadata.json"


class CamieTagger:
    """
    CamieTagger-V2 封装类：实现动漫风格图像的自动打标。
    支持自动下载模型、GPU加速推理以及按类别归纳标签。
    """

    def __init__(self, device: str = "cpu", cache_dir:  Union[str, Path, None] = None, local_only: bool = False):
        """
        初始化打标器。
        :param device: 推理设备，可选 "cpu" 或 "cuda"。
        :param cache_dir: 模型缓存目录。
        :param local_only: 是否只从本地读取，不连接 HuggingFace。
        """
        print(f"正在初始化 CamieTagger...")

        # 1. 下载或加载模型权重与元数据
        paths = self._prepare_files(cache_dir, local_only)

        # 2. 解析元数据
        self.metadata = self._load_json(paths['metadata'])
        self._parse_metadata()  # 预处理标签映射，提升后续查询速度

        # 3. 初始化 ONNX 推理会话
        self._init_session(paths['model'], device)

        # 4. 定义标准 ImageNet 归一化转换 (模型训练时使用的标准)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print("CamieTagger 初始化完成。")

    @staticmethod
    def _prepare_files(cache_dir, local_only):
        """从 HF 下载必要文件"""
        files = {
            'model': MODEL_FILE_NAME,
            'metadata': METADATA_FILE_NAME
        }
        return {
            key: hf_hub_download(
                repo_id=MODEL_REPO,
                filename=name,
                cache_dir=cache_dir,
                local_files_only=local_only
            ) for key, name in files.items()
        }

    @staticmethod
    def _load_json(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"解析元数据失败: {e}")

    @staticmethod
    def print_results(results: Dict[str, List[Dict]]):
        """打印打标结果的辅助函数"""
        for cat in results:
            print(f"\n[{cat.upper()}]")
            for item in results[cat]:
                print(f"\t{item['tag']}: {item['confidence']:.2%}")

    def _parse_metadata(self):
        """提取元数据中的关键映射关系，避免推理时重复解析"""
        ds_info = self.metadata['dataset_info']
        mapping = ds_info['tag_mapping']

        self.idx_to_tag = mapping['idx_to_tag']
        self.tag_to_category = mapping['tag_to_category']
        self.img_size = self.metadata['model_info']['img_size']

    def _init_session(self, model_path, device):
        """配置并启动 ONNX Runtime"""
        providers = []
        if ort.get_device() == 'GPU' and device.lower() == 'cuda':
            providers.append('CUDAExecutionProvider')
        elif device.lower() == 'cpu':
            providers.append('CPUExecutionProvider')
        else:
            raise ValueError(f"不支持的设备类型: {device}. 可选 'cpu' 或 'cuda'.")

        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            print(f"ONNX 会话已启动，当前 Provider: {self.session.get_providers()[0]}")
        except Exception as e:
            raise RuntimeError(f"创建 ONNX 会话失败，请检查环境: {e}")

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        核心图像预处理：等比例缩放 -> 补丁填充 (Padding) -> 归一化。
        :param image: PIL Image 对象
        :return: 推理所需的 4D numpy 数组 (1, C, H, W)
        """
        # 统一转为 RGB
        if image.mode != 'RGB':
            if image.mode == 'RGBA':
                # 创建一个纯色背景
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])  # 使用 alpha 通道作为掩码粘贴
                image = background
            else:
                image = image.convert('RGB')

        # 等比例缩放逻辑
        w, h = image.size
        ratio = w / h
        if ratio > 1:
            new_w = self.img_size
            new_h = int(new_w / ratio)
        else:
            new_h = self.img_size
            new_w = int(new_h * ratio)

        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 填充到模型要求的正方形尺寸 (使用 ImageNet 平均值背景色)
        pad_color = (124, 116, 104)
        new_img = Image.new('RGB', (self.img_size, self.img_size), pad_color)
        new_img.paste(image, ((self.img_size - new_w) // 2, (self.img_size - new_h) // 2))

        # 归一化并增加 Batch 维度
        img_tensor = self.transform(new_img)
        return img_tensor.unsqueeze(0).numpy()

    def tag(self, image: Image.Image, threshold: float = 0.61, top_k: int = 50) -> Dict[str, List[Dict]]:
        """
        执行打标。
        :param image: PIL Image 对象。
        :param threshold: 置信度阈值 (0-1)。
        :param top_k: 每个类别保留的前 K 个标签。
        :return: 包含分类标签及其置信度的字典。
        """
        print("开始推理...")

        # 加载图片
        if not isinstance(image, Image.Image):
            raise TypeError("参数 'image' 必须是 PIL.Image.Image 类型。请在传入前使用 Image.open() 加载图片。")

        # 1. 预处理
        input_data = self._preprocess_image(image)

        # 2. 推理
        start = time.time()
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: input_data})
        latency = time.time() - start

        # 3. 后处理逻辑
        # v2 模型通常有两个输出：[0] 是初始预测，[1] 是优化后的预测
        logits = outputs[1] if len(outputs) >= 2 else outputs[0]
        probs = 1.0 / (1.0 + np.exp(-logits[0]))  # Sigmoid 激活

        # 筛选与归类
        tags_by_cat = defaultdict(list)
        indices = np.where(probs >= threshold)[0]

        # 如果没有任何标签超过阈值，保底返回概率最高的 5 个
        if len(indices) == 0:
            indices = np.argsort(probs)[-5:][::-1]

        for idx in indices:
            idx_str = str(idx)
            tag_name = self.idx_to_tag.get(idx_str, f"unknown-{idx}")
            category = self.tag_to_category.get(tag_name, "general")
            conf = float(probs[idx])
            tags_by_cat[category].append({"tag": tag_name, "confidence": conf})

        # 排序并截断
        for cat in tags_by_cat:
            tags_by_cat[cat] = sorted(tags_by_cat[cat], key=lambda x: x['confidence'], reverse=True)[:top_k]

        print(f"推理完成，耗时: {latency:.4f}s, 检测到标签总数: {len(indices)}")
        return dict(tags_by_cat)
