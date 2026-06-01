# PreprocessAgent

`PreprocessAgent` is the first agent in the EvoMedSAM3 multi-agent workflow.
It performs image quality control, basic endoscopy image normalization, and an
optional MONAI-backed medical image analysis step before the image is sent to
the bbox prompt agent.

For 2D colonoscopy images, `MonaiTransformerTool` mounts
`PolypSegmentation2DTool` internally and loads
`andreribeiro87/unet3plus-efficientnet-kvasir-seg` from Hugging Face by
default. This is a Kvasir-SEG pretrained UNet3+ model with an EfficientNet-B0
backbone. Its probability map is the preferred source for
`candidate_region_hint`.

## Role in the Workflow

The agent protects the downstream pipeline from poor input. It checks whether
an uploaded image can be decoded, whether its size is reasonable, and whether
it is too dark, overexposed, blurry, or low contrast. Valid images are converted
to RGB, resized, contrast-enhanced, lightly denoised, and normalized to
`float32` in `[0, 1]`.

## MONAI Transformer Tool

`MonaiTransformerTool` is not the final segmentation model. It is a helper tool
called by the preprocessing agent to produce early medical image cues:

- `image_quality_score`
- `suspicious_heatmap`
- `semantic_embedding`
- `candidate_region_hint`

If MONAI is installed, the tool uses real MONAI transforms, then calls the 2D
polyp segmentation model inside the same tool. The SwinViT branch is deprecated
for this preprocessing workflow and is not loaded by default, even if
`model_swinvit.pt` exists next to this module.

## Handoff to the Bbox Prompt Agent

The bbox prompt agent can consume `candidate_region_hint["bbox"]`,
`suspicious_heatmap`, and `semantic_embedding` to decide where EvoMedSAM3 should
focus. Low-quality images are marked as `warning` or `reject` so they do not
enter segmentation silently.

The top-level `candidate_region_hint` is selected inside `MonaiTransformerTool`:

1. `PolypSegmentation2DTool` probability-map bbox.
2. MONAI-preprocessed classical saliency bbox as fallback.
3. No bbox, with manual review or rejection depending on quality status.

## Example

```powershell
Push-Location .\agent
& ..\.venv\Scripts\python.exe -m preprocess_agent.preprocess_agent ..\DemoAssets\example.png
Pop-Location
```

Disable the mounted 2D polyp model when testing only the MONAI saliency fallback:

```powershell
Push-Location .\agent
& ..\.venv\Scripts\python.exe -m preprocess_agent.preprocess_agent ..\DemoAssets\example.png --disable-2d-polyp-model
Pop-Location
```
