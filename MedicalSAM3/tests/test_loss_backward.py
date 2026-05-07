import unittest

import torch

from MedicalSAM3.exemplar.losses import (
    BoundaryBandDiceLoss,
    ExemplarConsistencyLoss,
    ExemplarInfoNCELoss,
    NegativeSuppressionLoss,
    PrototypeVarianceLoss,
    SoftHausdorffLoss,
)


class TestLossBackward(unittest.TestCase):
    def test_losses_are_scalar_and_backward(self) -> None:
        anchor = torch.randn(2, 8, requires_grad=True)
        positive = torch.randn(2, 8, requires_grad=True)
        negatives = torch.randn(2, 3, 8, requires_grad=True)
        logits_a = torch.randn(2, 1, 32, 32, requires_grad=True)
        logits_b = torch.randn(2, 1, 32, 32, requires_grad=True)
        target = torch.rand(2, 1, 32, 32)
        embeddings = torch.randn(4, 8, requires_grad=True)
        prototype = torch.randn(8, requires_grad=True)

        losses = [
            ExemplarInfoNCELoss()(anchor, positive, negatives),
            NegativeSuppressionLoss()(logits_a),
            ExemplarConsistencyLoss()(logits_a, logits_b),
            PrototypeVarianceLoss()(embeddings, prototype),
            BoundaryBandDiceLoss()(logits_a, target),
            SoftHausdorffLoss()(logits_a, target),
        ]
        total = sum(losses)
        self.assertEqual(total.dim(), 0)
        self.assertTrue(torch.isfinite(total).item())
        total.backward()
        self.assertIsNotNone(anchor.grad)
        self.assertIsNotNone(logits_a.grad)


if __name__ == "__main__":
    unittest.main()