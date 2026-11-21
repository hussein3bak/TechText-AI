"""
TechText AI Web Scraping Package
Handles ethical, rate-limited collection of technology-related textual data.
"""

from .robots import is_scraping_allowed
from .collector import TechScraper

__all__ = ["is_scraping_allowed", "TechScraper"]
