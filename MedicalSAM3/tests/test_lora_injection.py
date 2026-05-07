import unittest

import torch
import torch.nn as nn

from MedicalSAM3.adapters.lora import LoRAConfig, LoRALinear, apply_lora_to_model, mark_only_lora_as_trainable


class DummyAttentionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v_proj(torch.relu(self.q_proj(x)))


class DummyLoRAModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_encoder = nn.Module()
        self.image_encoder.blocks = nn.ModuleList([DummyAttentionBlock(), DummyAttentionBlock(), DummyAttentionBlock()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.image_encoder.blocks:
            x = block(x)
        return x


class TestLoRAInjection(unittest.TestCase):
    def test_lora_replaces_target_modules_and_backward(self) -> None:
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
