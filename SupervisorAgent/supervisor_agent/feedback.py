from __future__ import annotations

from .models import Feedback


class FeedbackCollector:
    def collect(self, feedback: Feedback) -> None:
        raise NotImplementedError


class LabelingExporter:
    def export(self, feedback: Feedback) -> None:
        raise NotImplementedError
