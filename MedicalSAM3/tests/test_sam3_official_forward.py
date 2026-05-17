import unittest

import torch

from MedicalSAM3.sam3_official.build_model import build_official_sam3_image_model
from MedicalSAM3.sam3_official.tensor_forward import Sam3TensorForwardWrapper


class TestSam3OfficialForward(unittest.TestCase):
    def test_sam3_official_forward_returns_mask_logits(self) -> None:
        model = build_official_sam3_image_model(
            checkpoint_path=None,
            device="cpu",
            dtype="fp32",
            compile_model=False,
            allow_dummy_fallback=True,
        )
        wrapper = Sam3TensorForwardWrapper(model=model, device="cpu", dtype="fp32")
        images = torch.rand(1, 3, 64, 64)
        outputs = wrapper(images=images, boxes=torch.tensor([[8.0, 8.0, 48.0, 48.0]]))

        self.assertIn("mask_logits", outputs)
        self.assertEqual(outputs["mask_logits"].shape[0], 1)
        self.assertEqual(outputs["masks"].shape[-2:], outputs["mask_logits"].shape[-2:])

    def test_sam3_official_forward_accepts_exemplar_tokens(self) -> None:
        model = build_official_sam3_image_model(
            checkpoint_path=None,
            device="cpu",
            dtype="fp32",
            compile_model=False,
            allow_dummy_fallback=True,
        )
        wrapper = Sam3TensorForwardWrapper(model=model, device="cpu", dtype="fp32")
        images = torch.rand(1, 3, 64, 64)
        exemplar_dim = int(getattr(wrapper.model, "hidden_dim", getattr(wrapper.model, "embed_dim", 128)))
        exemplar_tokens = torch.rand(1, 4, exemplar_dim)
        outputs = wrapper(
            images=images,
            text_prompt=["polyp"],
            exemplar_prompt_tokens=exemplar_tokens,
        )

        self.assertEqual(outputs["mask_logits"].shape[0], 1)
        self.assertEqual(tuple(outputs["scores"].shape), (1, 1))

    def test_sam3_official_forward_accepts_retrieval_prior(self) -> None:
        model = build_official_sam3_image_model(
            checkpoint_path=None,
            device="cpu",
            dtype="fp32",
            compile_model=False,
            allow_dummy_fallback=True,
        )
        wrapper = Sam3TensorForwardWrapper(model=model, device="cpu", dtype="fp32")
        images = torch.rand(1, 3, 64, 64)
        hidden_dim = int(getattr(wrapper.model, "hidden_dim", getattr(wrapper.model, "embed_dim", 128)))
        outputs = wrapper(
            images=images,
            boxes=torch.tensor([[8.0, 8.0, 48.0, 48.0]]),
            retrieval_prior={
                "semantic_prototype": torch.randn(1, hidden_dim),
                "semantic_prototype_map": torch.randn(1, hidden_dim, 16, 16),
                "spatial_bias_map": torch.rand(1, 1, 16, 16),
            },
        )

        if "retrieval_prior" in outputs["intermediate_features"]:
            self.assertIsInstance(outputs["intermediate_features"]["retrieval_prior"], dict)
        self.assertEqual(outputs["mask_logits"].shape[0], 1)


if __name__ == "__main__":
    unittest.main()
