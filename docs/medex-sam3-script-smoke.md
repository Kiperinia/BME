# MedEx-SAM3 Script Smoke

Use this smoke runner to verify that the main train, test, eval, and artifact scripts still launch without reading large default splits.

## Ubuntu Server

```bash
cd /share/home/huafuchen01/huangwei/WangRuiFeng/EvoSAM3
conda run -n sam3wangruifeng python MedicalSAM3/scripts/smoke_medex_scripts.py \
  --output-dir MedicalSAM3/outputs/script_smoke \
  --suite quick
```

For the retrieval demo/report scripts as well:

```bash
cd /share/home/huafuchen01/huangwei/WangRuiFeng/EvoSAM3
conda run -n sam3wangruifeng python MedicalSAM3/scripts/smoke_medex_scripts.py \
  --output-dir MedicalSAM3/outputs/script_smoke_full \
  --suite full
```

## Safety Notes

- The runner creates tiny synthetic images, masks, split files, and per-image metrics under the requested `MedicalSAM3/outputs/...` directory.
- It passes explicit split paths to train/eval scripts, so existing large default split files are not used.
- It restores SAM3 preflight target-map artifacts after the run to avoid dirtying tracked generated files.
- Keep the output directory inside `/share/home/huafuchen01/huangwei/WangRuiFeng/EvoSAM3`.
- Do not kill unrelated processes. If a run must be stopped, stop only the Python process started by this command.

The summary is written to:

```text
MedicalSAM3/outputs/script_smoke/smoke_summary.json
```
