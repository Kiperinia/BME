from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.patient import Patient


class PatientRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def list_all(self) -> list[Patient]:
        statement = select(Patient).order_by(Patient.exam_date.desc(), Patient.patient_id.asc())
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_by_id(self, patient_id: str) -> Patient | None:
        statement = select(Patient).where(Patient.patient_id == patient_id)
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def upsert_many(self, patients: list[Patient]) -> None:
        for patient in patients:
            await self.session.merge(patient)
        await self.session.commit()