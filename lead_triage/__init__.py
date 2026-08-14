"""Lead-Triage System — modular, configurable, explainable.

    from lead_triage import load_leads, ScoringEngine
    records, report = load_leads("data/default_leads.csv")
    ScoringEngine().score_all(records)
"""
from .ingest import LeadRecord, Settings, load_leads, default_dataset_path
from .scoring import ScoringEngine, ScoringConfig
from .nlp import IntentExtractor, IntentProfile, extract_intent
from . import core

__version__ = "1.0.0"
__all__ = [
    "LeadRecord", "Settings", "load_leads", "default_dataset_path",
    "ScoringEngine", "ScoringConfig",
    "IntentExtractor", "IntentProfile", "extract_intent", "core",
]
