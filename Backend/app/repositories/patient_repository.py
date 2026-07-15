from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.patient import Patient


class PatientRepository:
    """brief:
        Represent PatientRepository state and behavior.

    parameter:
        - session: Input value for session.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    def __init__(self, session: AsyncSession):
        """brief:
            Initialize this object.

        parameter:
            - session: Input value for session.

        retrival:
            - Returns None; performs side effects described in the brief section.
        """
        self.session = session

    async def list_all(self) -> list[Patient]:
        """brief:
            List all.

        parameter:
            - None.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        statement = select(Patient).order_by(Patient.exam_date.desc(), Patient.patient_id.asc())
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_by_id(self, patient_id: str) -> Patient | None:
        """brief:
            Get by id.

        parameter:
            - patient_id: Input value for patient_id.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        statement = select(Patient).where(Patient.patient_id == patient_id)
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def upsert_many(self, patients: list[Patient]) -> None:
        """brief:
            Handle upsert many.

        parameter:
            - patients: Input value for patients.

        retrival:
            - Returns None; performs side effects described in the brief section.
        """
        for patient in patients:
            await self.session.merge(patient)
        await self.session.commit()
