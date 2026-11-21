"""
TechText AI - Reusable Web Scraping & Data Collection Pipeline
Collects technology and artificial intelligence articles for model training.
Uses Wikipedia's official Content API (100% compliant with robots.txt & Wikimedia policies)
and provides general web scraping with robots.txt compliance.
"""

import os
import json
import time
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Set
import requests
from bs4 import BeautifulSoup

from .robots import is_scraping_allowed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Curated list of high-quality Technology and AI topics
DEFAULT_TECH_TOPICS = [
    "Artificial_intelligence",
    "Machine_learning",
    "Deep_learning",
    "Transformer_(deep_learning_architecture)",
    "Natural_language_processing",
    "Large_language_model",
    "Generative_artificial_intelligence",
    "Neural_network_(machine_learning)",
    "Prompt_engineering",
    "Chatbot",
    "Computer_vision",
    "Reinforcement_learning",
    "Speech_recognition",
    "Supervised_learning",
    "Unsupervised_learning",
    "Cloud_computing",
    "Graphics_processing_unit",
    "Software_engineering",
    "Algorithm",
    "Data_science",
    "Quantum_computing",
    "Cybersecurity",
    "Distributed_computing",
    "Robotics",
    "Autonomous_car",
    "Convolutional_neural_network",
    "Recurrent_neural_network",
    "OpenAI",
    "Turing_test",
    "Backpropagation",
]



class TechScraper:
    def __init__(
        self,
        output_dir: str = "data/raw",
        user_agent: str = "TechTextAI/1.0 (academic research; contact@buildfellowship.com)",
        request_delay: float = 0.5,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.user_agent = user_agent
        self.request_delay = request_delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml",
        })
        self.seen_urls: Set[str] = set()
        self.jsonl_path = self.output_dir / "tech_articles.jsonl"
        self.corpus_path = self.output_dir / "tech_corpus.txt"
        self._load_existing_urls()

    def _load_existing_urls(self):
        """Loads already scraped URLs to prevent duplicate requests."""
        if self.jsonl_path.exists():
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            if "url" in record:
                                self.seen_urls.add(record["url"])
                        except json.JSONDecodeError:
                            continue
        logger.info(f"Loaded {len(self.seen_urls)} existing URLs from {self.jsonl_path}")

    def fetch_url(self, url: str, check_robots: bool = True, max_retries: int = 3) -> Optional[str]:
        """Fetches HTML with retry logic, polite rate limiting, and optional robots.txt check."""
        if check_robots and not is_scraping_allowed(url, self.user_agent):
            logger.warning(f"robots.txt disallows scraping: {url}")
            return None

        for attempt in range(1, max_retries + 1):
            try:
                time.sleep(self.request_delay)
                response = self.session.get(url, timeout=15)
                if response.status_code == 200:
                    return response.text
                elif response.status_code == 429:
                    wait_time = attempt * 4
                    logger.warning(f"Rate limited (429). Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt}/{max_retries} failed for {url}: {e}")
                time.sleep(attempt * 2)

        return None

    def parse_article(self, url: str, html: str, default_title: str = "Untitled Tech Article") -> Optional[Dict]:
        """Parses article content, removing navigation, scripts, ads, and references."""
        soup = BeautifulSoup(html, "html.parser")

        # Extract title
        title_tag = soup.find("h1") or soup.find("title")
        title = title_tag.get_text().strip() if title_tag else default_title

        # Remove unwanted elements
        for unwanted in soup(
            ["script", "style", "nav", "footer", "header", "aside", "noscript", "table", "sup", "link", "meta", "figure"]
        ):
            unwanted.decompose()

        # Extract all paragraphs across all sections
        paragraphs = []
        for p in soup.find_all("p"):
            text = p.get_text().strip()
            # Filter out tiny boilerplate / empty lines
            if len(text) > 40:
                paragraphs.append(text)

        full_text = "\n\n".join(paragraphs)
        if len(full_text) < 150:
            return None

        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        words = full_text.split()

        return {
            "id": url_hash,
            "title": title,
            "url": url,
            "collection_date": datetime.now(timezone.utc).isoformat(),
            "paragraph_count": len(paragraphs),
            "word_count": len(words),
            "text": full_text,
        }

    def scrape_topic(self, topic: str) -> Optional[Dict]:
        """Fetches and parses a tech topic via Wikipedia's public REST content API."""
        canonical_url = f"https://en.wikipedia.org/wiki/{topic}"
        if canonical_url in self.seen_urls:
            logger.info(f"Skipping already collected topic: {topic}")
            return None

        api_url = f"https://en.wikipedia.org/api/rest_v1/page/html/{topic}"
        logger.info(f"Fetching topic '{topic}' via Wikimedia Content API...")

        # Wikimedia REST API explicitly permits automated retrieval with a descriptive User-Agent
        html = self.fetch_url(api_url, check_robots=False)
        if not html:
            return None

        readable_title = topic.replace("_", " ")
        article = self.parse_article(canonical_url, html, default_title=readable_title)
        if article:
            self._save_article(article)
            self.seen_urls.add(canonical_url)
            logger.info(f"Successfully collected '{article['title']}' ({article['word_count']} words)")
            return article
        return None

    def _save_article(self, article: Dict):
        """Appends article record to JSONL and corpus text file."""
        with open(self.jsonl_path, "a", encoding="utf-8") as f_jsonl:
            f_jsonl.write(json.dumps(article, ensure_ascii=False) + "\n")

        with open(self.corpus_path, "a", encoding="utf-8") as f_txt:
            f_txt.write(f"\n\n--- ARTICLE: {article['title']} ---\n")
            f_txt.write(article["text"])
            f_txt.write("\n")

    def run(self, topics: Optional[List[str]] = None, max_articles: int = 15) -> List[Dict]:
        """Executes scraping across the given topic list up to max_articles."""
        topic_list = topics or DEFAULT_TECH_TOPICS
        collected = []
        count = 0

        for topic in topic_list:
            if count >= max_articles:
                break
            article = self.scrape_topic(topic)
            if article:
                collected.append(article)
                count += 1

        logger.info(f"Scraping completed! Collected {len(collected)} new tech articles.")
        return collected


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TechText AI Scraper")
    parser.add_argument("--limit", type=int, default=15, help="Maximum articles to collect")
    parser.add_argument("--delay", type=float, default=0.3, help="Seconds delay between requests")
    args = parser.parse_args()

    scraper = TechScraper(request_delay=args.delay)
    scraper.run(max_articles=args.limit)


if __name__ == "__main__":
    main()
