"""
TechText AI - Robots.txt Checker
Ensures ethical web scraping by respecting website robots.txt policies.
"""

from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Cache of RobotFileParsers per netloc (domain)
_ROBOTS_CACHE: Dict[str, RobotFileParser] = {}


def get_robots_parser(url: str, user_agent: str = "TechTextAI-ResearchBot/1.0") -> RobotFileParser:
    """
    Fetches and parses the robots.txt file for a given URL's domain.
    Caches the parser to prevent redundant network requests.
    """
    parsed = urlparse(url)
    domain = f"{parsed.scheme}://{parsed.netloc}"

    if domain in _ROBOTS_CACHE:
        return _ROBOTS_CACHE[domain]

    robots_url = f"{domain}/robots.txt"
    parser = RobotFileParser()
    parser.set_url(robots_url)
    try:
        parser.read()
    except Exception as e:
        logger.warning(f"Could not read robots.txt from {robots_url}: {e}. Defaulting to cautious crawl.")
        # If robots.txt cannot be fetched, we allow crawling if no disallow exists
    _ROBOTS_CACHE[domain] = parser
    return parser


def is_scraping_allowed(url: str, user_agent: str = "TechTextAI-ResearchBot/1.0") -> bool:
    """
    Determines whether scraping the given URL is permitted for user_agent.
    """
    try:
        parser = get_robots_parser(url, user_agent=user_agent)
        return parser.can_fetch(user_agent, url)
    except Exception as e:
        logger.warning(f"Error checking robots permission for {url}: {e}")
        return True
