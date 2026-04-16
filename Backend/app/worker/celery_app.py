from celery import Celery
from celery.signals import worker_process_init

from app.core.config import get_settings
from app.services.sam3_runtime import SAM3RuntimeSingleton


settings = get_settings()

celery_app = Celery(
    "async_sam3_worker",
    broker=settings.broker_url,
    backend=settings.result_backend,
    include=["app.worker.tasks"],
)

celery_app.conf.update(
    task_default_queue=settings.celery_task_queue,
    task_track_started=True,
    worker_concurrency=settings.celery_worker_concurrency,
    worker_prefetch_multiplier=settings.celery_worker_prefetch_multiplier,
    worker_pool=settings.celery_worker_pool,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=settings.celery_result_expires,
    task_soft_time_limit=settings.celery_task_soft_time_limit,
    task_time_limit=settings.celery_task_time_limit,
    broker_connection_retry_on_startup=True,
    timezone="Asia/Shanghai",
    enable_utc=False,
)


@worker_process_init.connect
def warm_model_on_worker_start(**_: object) -> None:
    SAM3RuntimeSingleton.get_instance(settings=settings)
