from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class AppException(Exception):
    """brief:
        Represent AppException state and behavior.

    parameter:
        - status_code: Input value for status_code.
        - error_code: Input value for error_code.
        - message: Input value for message.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    def __init__(self, status_code: int, error_code: int, message: str):
        """brief:
            Initialize this object.

        parameter:
            - status_code: Input value for status_code.
            - error_code: Input value for error_code.
            - message: Input value for message.

        retrival:
            - Returns None; performs side effects described in the brief section.
        """
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        super().__init__(message)


def build_http_exception(status_code: int, error_code: int, message: str) -> HTTPException:
    """brief:
        Build http exception.

    parameter:
        - status_code: Input value for status_code.
        - error_code: Input value for error_code.
        - message: Input value for message.

    retrival:
        - Returns the computed value for the caller or workflow.
    """
    return HTTPException(
        status_code=status_code,
        detail={"code": error_code, "message": message},
    )


async def app_exception_handler(_: Request, exc: AppException) -> JSONResponse:
    """brief:
        Handle app exception handler.

    parameter:
        - _: Input value for _.
        - exc: Input value for exc.

    retrival:
        - Returns the computed value for the caller or workflow.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.error_code, "message": exc.message, "data": None},
    )


async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    """brief:
        Handle http exception handler.

    parameter:
        - _: Input value for _.
        - exc: Input value for exc.

    retrival:
        - Returns the computed value for the caller or workflow.
    """
    detail = exc.detail
    if isinstance(detail, dict):
        code = detail.get("code", exc.status_code)
        message = detail.get("message", "request failed")
    else:
        code = exc.status_code
        message = str(detail)

    return JSONResponse(
        status_code=exc.status_code,
        content={"code": code, "message": message, "data": None},
    )


async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """brief:
        Handle validation exception handler.

    parameter:
        - _: Input value for _.
        - exc: Input value for exc.

    retrival:
        - Returns the computed value for the caller or workflow.
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "code": 42200,
            "message": "request validation failed",
            "data": {"errors": exc.errors()},
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """brief:
        Register exception handlers.

    parameter:
        - app: Input value for app.

    retrival:
        - Returns None; performs side effects described in the brief section.
    """
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
