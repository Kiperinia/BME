# AGENTS.md

## Core Working Rules

- Think before coding: state assumptions, surface ambiguity, and ask when a decision changes the implementation.
- Prefer the smallest solution that satisfies the request. Avoid speculative abstractions, extra configurability, and impossible-scenario handling.
- Make surgical edits. Match local style, touch only the files and lines needed, and only remove dead code created by your own change.
- Define a concrete verification target before editing. For multi-step work, write a short plan with a check for each step, then run the narrowest useful validation after changes.

## Repo Map

- Backend/: FastAPI + Celery SAM3 backend. Keep the existing api/core/models/repositories/schemas/services/worker split. Read [Backend/README.md](Backend/README.md) before changing API or worker behavior.
- Frontend/: Vue 3 + Vite + TypeScript report builder UI. Use [Frontend/docs/frontend-structure-guide.md](Frontend/docs/frontend-structure-guide.md) for file placement and [Frontend/docs/report-builder-spec.md](Frontend/docs/report-builder-spec.md) for UI and API contracts.
- agent/: hello_agents-based diagnosis pipeline. Main entry is [agent/run_minimal_agent.py](agent/run_minimal_agent.py). LLM profiles live in [agent/config/llm_profiles.json](agent/config/llm_profiles.json).
- MedicalSAM3/: training, validation, and runtime wrappers. Run [check_sam3_import.py](check_sam3_import.py) before changing SAM3 runtime code on Windows.
- DataSetTrans/: dataset conversion utilities. Avoid format changes unless the task is explicitly about dataset transformation.

## Practical Commands

- Root venv: .\\.venv\\Scripts\\Activate.ps1
- Frontend dev: cd Frontend; npm install; npm run dev
- Frontend build: cd Frontend; npm run build
- Backend API: cd Backend; pip install -r requirements.txt; uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
- Backend worker on Windows: cd Backend; celery -A app.worker.celery_app:celery_app worker --loglevel=info --pool=solo --concurrency=1
- Agent smoke run: Push-Location .\\agent; & ..\\.venv\\Scripts\\Activate.ps1; python .\\run_minimal_agent.py; Pop-Location
- SAM3 import check: python .\\check_sam3_import.py

## Repo-Specific Pitfalls

- This workspace is developed on Windows. SAM3 depends on the local Triton wheel at [triton-3.0.0-cp312-cp312-win_amd64.whl](triton-3.0.0-cp312-cp312-win_amd64.whl), and import failures may come from missing extras such as pandas rather than a missing sam3 package.
- Backend workers must stay on the solo pool with concurrency 1 on Windows.
- Backend currently supports a mock-friendly SAM3 path for frontend integration. Preserve public response contracts unless the task is specifically about changing the API.
- Frontend already has a documented folder split. Prefer existing areas such as src/api, components/common, components/report, stores, and types instead of inventing new structure.
- Avoid editing large generated artifacts under MedicalSAM3/checkpoint, MedicalSAM3/data, MedicalSAM3/outputs, and agent/memory/traces unless the task explicitly targets them.

## Validation Guidance

- There is no obvious single repo-wide test command. Use the smallest relevant check for the area you changed.
- Frontend: prefer npm run build for type and bundle validation.
- Backend: prefer the narrowest startup or endpoint smoke check that exercises the changed slice.
- agent/MedicalSAM3: prefer targeted import checks or the smallest script run that covers the touched path.
- If you could not run validation, say so explicitly and state what remains unverified.