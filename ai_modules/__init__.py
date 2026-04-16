"""AI Modules Package"""

from ai_modules.gemini_api import GeminiHealthAssistant, get_gemini_assistant

__all__ = [
    'HealthRiskPredictor',
    'GeminiHealthAssistant',
    'get_gemini_assistant'
]


def __getattr__(name):
    if name == 'HealthRiskPredictor':
        from app.ml.predictor import HealthRiskPredictor
        return HealthRiskPredictor
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
