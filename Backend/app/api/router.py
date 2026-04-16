from fastapi import APIRouter

from app.api.endpoints.analysis import router as analysis_router
from app.api.endpoints.sam3_inference import router as sam3_inference_router


api_router = APIRouter()
api_router.include_router(analysis_router, prefix="/analysis", tags=["analysis"])
api_router.include_router(sam3_inference_router, prefix="/analysis", tags=["analysis"])
