#!/usr/bin/env python3
"""
Scrape GitHub for repositories using filterpy library.

This script searches GitHub for repositories that import or use the filterpy
library and appends the discoveries to a CSV file. It handles pagination,
deduplication, and rate limiting according to GitHub API guidelines.
"""

import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

from tqdm import tqdm


class GitHubSearchError(Exception):
    """Exception raised for GitHub API search errors."""

    pass


class FilterPyRepoScraper:
    """Scrapes GitHub for repositories using the filterpy library."""

    def __init__(self, output_file: Path = Path("scripts/filterpy_repos.csv")) -> None:
        """
        Initialize the scraper.

        Args:
            output_file: Path to the CSV file for storing results.
        """
        self.output_file = output_file
        self.csv_columns = [
            "repo_full_name",
            "repo_url",
            "stars",
            "forks",
            "description",
            "last_updated",
            "primary_language",
            "discovered_at",
        ]
        self._ensure_csv_exists()

    def _ensure_csv_exists(self) -> None:
        """Create CSV file with headers if it doesn't exist."""
        if not self.output_file.exists():
            with open(self.output_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_columns)
                writer.writeheader()

    def _get_existing_repos(self) -> set[str]:
        """
        Get set of existing repository full names.

        Returns:
            Set of repository full names already in the CSV.
        """
        existing = set()
        if self.output_file.exists():
            with open(self.output_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["repo_full_name"]:
                        existing.add(row["repo_full_name"])
        return existing

    def _call_gh_api(self, endpoint: str) -> dict[str, Any]:
        """
        Call GitHub API using gh cli.

        Args:
            endpoint: API endpoint path with query parameters.

        Returns:
            Parsed JSON response.

        Raises:
            GitHubSearchError: If the API call fails.
        """
        cmd = ["gh", "api", endpoint, "--method", "GET"]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            raise GitHubSearchError(f"GitHub API call failed: {e.stderr}") from e
        except json.JSONDecodeError as e:
            raise GitHubSearchError(f"Failed to parse API response: {e}") from e

    def _search_code(self, search_pattern: str, page: int = 1, per_page: int = 100) -> dict[str, Any]:
        """
        Search for code matching a pattern.

        Args:
            search_pattern: Code search pattern (e.g., "import filterpy language:python").
            page: Page number (1-indexed).
            per_page: Results per page (max 100).

        Returns:
            API response containing search results and metadata.
        """
        # Properly escape the search pattern for the URL
        encoded_q = quote(search_pattern)
        endpoint = f"/search/code?q={encoded_q}&page={page}&per_page={min(per_page, 100)}"
        return self._call_gh_api(endpoint)

    def _get_repo_details(self, owner: str, repo: str) -> Optional[dict[str, Any]]:
        """
        Get detailed information about a repository.

        Args:
            owner: Repository owner.
            repo: Repository name.

        Returns:
            Repository details or None if API call fails.
        """
        try:
            endpoint = f"/repos/{owner}/{repo}"
            result = self._call_gh_api(endpoint)
            return {
                "repo_full_name": result.get("full_name", ""),
                "repo_url": result.get("html_url", ""),
                "stars": result.get("stargazers_count", 0),
                "forks": result.get("forks_count", 0),
                "description": result.get("description") or "",
                "last_updated": result.get("updated_at", ""),
                "primary_language": result.get("language") or "Unknown",
                "discovered_at": datetime.now().isoformat(),
            }
        except GitHubSearchError:
            return None

    def _extract_repos_from_search(self, search_results: dict[str, Any]) -> list[tuple[str, str]]:
        """
        Extract unique (owner, repo) tuples from search results.

        Args:
            search_results: API response from code search.

        Returns:
            List of (owner, repo) tuples.
        """
        repos = []
        items = search_results.get("items", [])

        for item in items:
            repo_path = item.get("repository", {}).get("full_name", "")
            if repo_path and "/" in repo_path:
                owner, repo = repo_path.rsplit("/", 1)
                repos.append((owner, repo))

        # Remove duplicates while preserving order
        seen = set()
        unique_repos = []
        for owner, repo in repos:
            if (owner, repo) not in seen:
                seen.add((owner, repo))
                unique_repos.append((owner, repo))

        return unique_repos

    def _append_to_csv(self, repo_data: dict[str, Any]) -> None:
        """
        Append a repository entry to the CSV file.

        Args:
            repo_data: Dictionary containing repository information.
        """
        with open(self.output_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_columns)
            writer.writerow(repo_data)

    def scrape(self) -> int:
        """
        Scrape GitHub for filterpy repositories.

        Performs searches for both "import filterpy" and "from filterpy"
        patterns in Python code, handles pagination, and appends new
        discoveries to the CSV file.

        Returns:
            Number of new repositories discovered.
        """
        search_patterns = [
            "import filterpy language:python",
            "from filterpy language:python",
        ]

        existing_repos = self._get_existing_repos()
        new_count = 0

        for pattern in search_patterns:
            print(f"\nSearching for: '{pattern}'")

            page = 1
            while True:
                try:
                    results = self._search_code(pattern, page=page)
                    items = results.get("items", [])

                    if not items:
                        break

                    repos = self._extract_repos_from_search(results)

                    pbar = tqdm(
                        repos,
                        desc=f"Processing results (page {page})",
                        unit="repo",
                    )

                    for owner, repo in pbar:
                        repo_full_name = f"{owner}/{repo}"

                        if repo_full_name not in existing_repos:
                            details = self._get_repo_details(owner, repo)
                            if details:
                                self._append_to_csv(details)
                                existing_repos.add(repo_full_name)
                                new_count += 1

                    # Check if there are more pages
                    # GitHub code search has a limit of ~1000 results
                    if len(items) < 100 or page >= 10:
                        break

                    page += 1

                except GitHubSearchError as e:
                    print(f"Error during search: {e}", file=sys.stderr)
                    break

        return new_count


def main() -> int:
    """
    Main entry point.

    Returns:
        Exit code (0 for success).
    """
    try:
        scraper = FilterPyRepoScraper()
        new_repos = scraper.scrape()
        print(f"\n✓ Scraping complete. Found {new_repos} new repositories.")
        return 0
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
