# Ubuntu Server Codex Guardrails

This document is for Codex sessions running on the shared Ubuntu server.

## Fixed Environment

- Project root: `/share/home/huafuchen01/huangwei/WangRuiFeng/EvoSAM3`
- Conda environment: `sam3wangruifeng`
- Default shell setup:

```bash
cd /share/home/huafuchen01/huangwei/WangRuiFeng/EvoSAM3
conda activate sam3wangruifeng
```

When a non-interactive command is safer, use:

```bash
conda run -n sam3wangruifeng python check_sam3_import.py
```

## Hard Boundaries

- Work only inside `/share/home/huafuchen01/huangwei/WangRuiFeng/EvoSAM3`.
- Do not edit sibling directories under `/share/home/huafuchen01/huangwei/WangRuiFeng`.
- Do not edit global conda configuration, system Python, CUDA drivers, shared package caches, or shell startup files.
- Do not run `sudo`, global `pip install`, global `conda install`, or package-manager commands unless the user explicitly approves that exact operation.
- Do not remove shared datasets, checkpoints, `MedicalSAM3/data`, `MedicalSAM3/checkpoint`, `MedicalSAM3/outputs`, `memory/traces`, or other large generated artifacts unless the user explicitly names the artifact to remove.
- Do not kill unrelated processes. Before stopping any process, confirm it was started by the current task and record its PID, command, and working directory.
- Do not hide long-running terminal work. Prefer visible shell output or a named `tmux` session that the user can attach to.

## Linux Path Rules

- Use paths relative to the project root when possible, for example `MedicalSAM3/outputs/medex_sam3/...`.
- Use the absolute project root only for launcher docs, scheduler scripts, or environment examples.
- Replace old Windows paths such as `E:\BME\...` with either project-relative paths or `/share/home/huafuchen01/huangwei/WangRuiFeng/EvoSAM3/...`.
- Keep runtime outputs under `MedicalSAM3/outputs/...` or an explicitly requested output directory inside the project root.

## Dependency Rules

- Backend dependencies: `requirements/backend-requirements.txt`
- Agent dependencies: `requirements/agent-requirements.txt`
- Model/SAM3 dependencies: `requirements/model-requirements.txt`
- Frontend dependencies: `requirements/frontend-requirements.txt` plus `Frontend/package-lock.json`

Install Python packages only into `sam3wangruifeng`:

```bash
conda run -n sam3wangruifeng python -m pip install -r requirements/model-requirements.txt
```

For frontend work, install only in `Frontend/`:

```bash
cd /share/home/huafuchen01/huangwei/WangRuiFeng/EvoSAM3/Frontend
npm ci
```

## Verification Targets

- Environment: `conda run -n sam3wangruifeng python check_sam3_import.py`
- Backend syntax/import smoke: `cd Backend && conda run -n sam3wangruifeng python -m compileall app`
- Agent smoke: `cd agent && conda run -n sam3wangruifeng python run_minimal_agent.py`
- Model scripts syntax: `conda run -n sam3wangruifeng python -m compileall MedicalSAM3/scripts`
- Model train/test/eval script smoke: `conda run -n sam3wangruifeng python MedicalSAM3/scripts/smoke_medex_scripts.py --output-dir MedicalSAM3/outputs/script_smoke --suite quick`
- Frontend build: `cd Frontend && npm run build`

If a verification step needs data, checkpoints, Redis, MySQL, GPU time, or credentials that are unavailable, report the missing prerequisite instead of changing shared server state to force the run.
