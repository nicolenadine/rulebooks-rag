"""Ingestion module for loading and preprocessing parsed PDF data."""

from src.ingestion.mini_page_detector import MiniPageDetector
from src.ingestion.parse_loader import ParseLoader

__all__ = ["ParseLoader", "MiniPageDetector"]
