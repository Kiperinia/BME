import unittest

import torch
import torch.nn as nn

from MedicalSAM3.adapters.lora import LoRAConfig, LoRALinear, apply_lora_to_model, mark_only_lora_as_trainable


class DummyAttentionBlock(nn.Module):
    """测试辅助类，模拟包含 q_proj 和 v_proj 的注意力块。"""
    def __init__(self) -> None:
        """brief:
            Initialize this object.

        parameter:
            - None.

        retrival:
            - Returns None; performs side effects described in the brief section.
        """
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """brief:
            Handle forward.

        parameter:
            - x: Input value for x.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return self.v_proj(torch.relu(self.q_proj(x)))


class DummyLoRAModel(nn.Module):
    """测试辅助类，模拟包含多个注意力块的图像编码器模型。"""
    def __init__(self) -> None:
        """初始化模型，包含三个 DummyAttentionBlock。

        参数：
            - 无

        返回：
            - 无
        """
        super().__init__()
        self.image_encoder = nn.Module()
        self.image_encoder.blocks = nn.ModuleList([DummyAttentionBlock(), DummyAttentionBlock(), DummyAttentionBlock()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：依次通过所有注意力块。

        参数：
            - x: 输入张量

        返回：
            - 输出张量
        """
        for block in self.image_encoder.blocks:
            x = block(x)
        return x


class TestLoRAInjection(unittest.TestCase):
    """测试 LoRA 适配器的注入、训练标记和前向反向传播。"""
    def test_lora_replaces_target_modules_and_backward(self) -> None:
        """验证 LoRA 注入替换目标模块、正确标记可训练参数并支持反向传播。"""
        model = DummyLoRAModel()
        config = LoRAConfig(rank=4, alpha=8, dropout=0.0, target_scopes=["vision_encoder"])
        replaced = apply_lora_to_model(model, config)
        self.assertTrue(replaced)
        self.assertIsInstance(model.image_encoder.blocks[2].q_proj, LoRALinear)
        mark_only_lora_as_trainable(model)

        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all("lora_" in name for name in trainable))

        inputs = torch.randn(2, 4, 8)
        outputs = model(inputs)
        self.assertEqual(tuple(outputs.shape), (2, 4, 8))

        loss = outputs.sum()
        loss.backward()
        lora_modules = [module for module in model.modules() if isinstance(module, LoRALinear)]
        self.assertTrue(any(module.lora_A.weight.grad is not None for module in lora_modules))
        self.assertTrue(any(module.lora_B.weight.grad is not None for module in lora_modules))


if __name__ == "__main__":
    unittest.main()
