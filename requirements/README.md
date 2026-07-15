# Dependency Exports

This directory splits project dependencies by runtime area.

- `frontend-requirements.txt`: npm dependencies exported from `Frontend/package.json`.
- `backend-requirements.txt`: FastAPI, Celery, database, and backend SAM3 integration packages.
- `agent-requirements.txt`: hello-agents diagnosis pipeline and medical report tooling packages.
- `model-requirements.txt`: MedicalSAM3 training, validation, retrieval, and evaluation packages.

On the Ubuntu server, install Python dependencies only inside conda environment `sam3wangruifeng` and only while working under `/share/home/huafuchen01/huangwei/WangRuiFeng/EvoSAM3`.
