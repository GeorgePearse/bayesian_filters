#!/usr/bin/env python3
"""
Import comments from upstream issues to the corresponding fork issues.

This script fetches all comments from upstream rlabbe/filterpy issues and
posts them to the corresponding issues in the fork, with proper attribution.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


UPSTREAM_REPO = "rlabbe/filterpy"
FORK_REPO = "GeorgePearse/filterpy"
TRACKING_FILE = "imported-issues.json"
COMMENT_TRACKING_FILE = "imported-comments.json"
RATE_LIMIT_DELAY = 1.0  # seconds between comment creation


def run_gh_command(args: list[str]) -> str:
    """Run a gh CLI command and return the output."""
    try:
        result = subprocess.run(
            ["gh"] + args,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running gh command: {e}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        raise


def run_gh_api(endpoint: str) -> Any:
    """Run a gh API command and return parsed JSON."""
    try:
        result = subprocess.run(
            ["gh", "api", endpoint],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running gh API command: {e}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        raise


def load_tracking_data() -> dict[str, Any]:
    """Load the issue tracking file."""
    tracking_path = Path(TRACKING_FILE)
    if not tracking_path.exists():
        print(
            f"Error: {TRACKING_FILE} not found. Run import-upstream-issues.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(tracking_path) as f:
        return json.load(f)


def load_comment_tracking_data() -> dict[str, Any]:
    """Load the comment tracking file to see which comments have been imported."""
    tracking_path = Path(COMMENT_TRACKING_FILE)
    if tracking_path.exists():
        with open(tracking_path) as f:
            return json.load(f)
    return {"imported_comments": {}}


def save_comment_tracking_data(data: dict[str, Any]) -> None:
    """Save the comment tracking file."""
    with open(COMMENT_TRACKING_FILE, "w") as f:
        json.dump(data, f, indent=2)


def fetch_upstream_comments(issue_number: int) -> list[dict[str, Any]]:
    """Fetch all comments for an upstream issue."""
    comments = run_gh_api(f"repos/{UPSTREAM_REPO}/issues/{issue_number}/comments")
    return comments if isinstance(comments, list) else []


def extract_fork_issue_number(fork_url: str) -> int:
    """Extract the issue number from a GitHub issue URL."""
    # URL format: https://github.com/GeorgePearse/filterpy/issues/3
    return int(fork_url.rstrip("/").split("/")[-1])


def format_comment_body(original_comment: dict[str, Any]) -> str:
    """Format a comment with attribution to the original author."""
    author = original_comment["user"]["login"]
    created_at = datetime.fromisoformat(original_comment["created_at"].replace("Z", "+00:00"))
    original_url = original_comment["html_url"]
    body = original_comment["body"]

    formatted_parts = [
        f"**Original comment by @{author}** ([link]({original_url}))",
        f"**Posted:** {created_at.strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "---",
        "",
        body.strip(),
    ]

    return "\n".join(formatted_parts)


def post_comment(fork_issue_number: int, comment_body: str) -> None:
    """Post a comment to a fork issue."""
    run_gh_command(
        [
            "issue",
            "comment",
            str(fork_issue_number),
            "--repo",
            FORK_REPO,
            "--body",
            comment_body,
        ]
    )


def main() -> None:
    """Main entry point."""
    print("=" * 70)
    print("Import Upstream Issue Comments")
    print("=" * 70)
    print()

    # Load tracking data
    issue_tracking = load_tracking_data()
    comment_tracking = load_comment_tracking_data()
    imported_comments = comment_tracking.get("imported_comments", {})

    imported_issues = issue_tracking.get("imported", {})

    if not imported_issues:
        print("No imported issues found. Run import-upstream-issues.py first.")
        return

    print(f"Found {len(imported_issues)} imported issues")
    print()

    total_comments_imported = 0
    total_comments_skipped = 0
    issues_with_comments = 0

    for upstream_number, issue_info in sorted(imported_issues.items(), key=lambda x: int(x[0])):
        upstream_number_int = int(upstream_number)
        fork_url = issue_info["fork_url"]
        fork_issue_number = extract_fork_issue_number(fork_url)
        issue_title = issue_info["title"]

        print(f"Processing upstream issue #{upstream_number_int}: {issue_title[:50]}...")

        # Fetch comments from upstream
        try:
            comments = fetch_upstream_comments(upstream_number_int)
        except Exception as e:
            print(f"  ✗ Failed to fetch comments: {e}")
            continue

        if not comments:
            print("  ℹ No comments found")
            continue

        print(f"  Found {len(comments)} comment(s)")
        issues_with_comments += 1

        # Track comments for this issue
        if upstream_number not in imported_comments:
            imported_comments[upstream_number] = {}

        issue_comment_tracking = imported_comments[upstream_number]

        # Post each comment
        for idx, comment in enumerate(comments, 1):
            comment_id = str(comment["id"])

            # Skip if already imported
            if comment_id in issue_comment_tracking:
                print(f"    [{idx}/{len(comments)}] Comment {comment_id} already imported")
                total_comments_skipped += 1
                continue

            try:
                formatted_body = format_comment_body(comment)
                post_comment(fork_issue_number, formatted_body)

                # Track the import
                issue_comment_tracking[comment_id] = {
                    "imported_at": datetime.now().isoformat(),
                    "author": comment["user"]["login"],
                    "fork_issue_number": fork_issue_number,
                }

                print(f"    [{idx}/{len(comments)}] ✓ Imported comment by @{comment['user']['login']}")
                total_comments_imported += 1

                # Save after each comment
                comment_tracking["imported_comments"] = imported_comments
                save_comment_tracking_data(comment_tracking)

                # Rate limiting
                if idx < len(comments):
                    time.sleep(RATE_LIMIT_DELAY)

            except Exception as e:
                print(f"    [{idx}/{len(comments)}] ✗ Failed to import comment {comment_id}: {e}")
                continue

        print()

    print("=" * 70)
    print("Import Complete")
    print("=" * 70)
    print(f"Issues processed: {len(imported_issues)}")
    print(f"Issues with comments: {issues_with_comments}")
    print(f"Comments imported: {total_comments_imported}")
    print(f"Comments skipped (already imported): {total_comments_skipped}")
    print()
    print(f"Comment tracking data saved to: {COMMENT_TRACKING_FILE}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress has been saved.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}", file=sys.stderr)
        sys.exit(1)
