"""Reporting modules live here."""

from kubera.reporting.offline_evaluation import evaluate_offline

__all__ = ["evaluate_offline", "generate_final_review"]


def __getattr__(name: str):
    if name == "generate_final_review":
        from kubera.reporting.final_review import generate_final_review

        return generate_final_review
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
