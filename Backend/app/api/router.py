from fastapi import APIRouter

from app.api.endpoints.analysis import router as analysis_router
from app.api.endpoints.agent_workflow import router as agent_workflow_router
from app.api.endpoints.report_context import router as report_context_router
from app.api.endpoints.sam3_inference import router as sam3_inference_router
from app.api.endpoints.system_settings import router as system_settings_router


api_router = APIRouter()
api_router.include_router(analysis_router, prefix="/analysis", tags=["analysis"])
api_router.include_router(sam3_inference_router, prefix="/analysis", tags=["analysis"])
api_router.include_router(agent_workflow_router, tags=["agent-workflow"])
api_router.include_router(report_context_router, tags=["report-context"])
api_router.include_router(system_settings_router, tags=["system-settings"])
