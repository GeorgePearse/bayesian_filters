#!/usr/bin/env python3
"""
Import open issues from the original rlabbe/filterpy repository.

This script fetches all open issues from the upstream repository and creates
corresponding issues in the fork, with links back to the originals.
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
UPSTREAM_LABEL = "from-upstream"
TRACKING_FILE = "imported-issues.json"
RATE_LIMIT_DELAY = 1.5  # seconds between issue creation


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


def ensure_label_exists() -> None:
    """Ensure the 'from-upstream' label exists in the fork."""
    print(f"Ensuring '{UPSTREAM_LABEL}' label exists...")
    try:
        # Try to create the label
        run_gh_command(
            [
                "label",
                "create",
                UPSTREAM_LABEL,
                "--repo",
                FORK_REPO,
                "--description",
                "Issue imported from upstream rlabbe/filterpy repository",
                "--color",
                "0366d6",
            ]
        )
        print(f"✓ Created '{UPSTREAM_LABEL}' label")
    except subprocess.CalledProcessError:
        # Label might already exist
        print(f"✓ '{UPSTREAM_LABEL}' label already exists")


def fetch_upstream_issues() -> list[dict[str, Any]]:
    """Fetch all open issues from the upstream repository."""
    print(f"Fetching open issues from {UPSTREAM_REPO}...")
    output = run_gh_command(
        [
            "issue",
            "list",
            "--repo",
            UPSTREAM_REPO,
            "--state",
            "open",
            "--limit",
            "1000",
            "--json",
            "number,title,body,labels,createdAt,updatedAt,url",
        ]
    )
    issues = json.loads(output)
    print(f"✓ Found {len(issues)} open issues")
    return issues


def load_tracking_data() -> dict[str, Any]:
    """Load the tracking file to see which issues have already been imported."""
    tracking_path = Path(TRACKING_FILE)
    if tracking_path.exists():
        with open(tracking_path) as f:
            return json.load(f)
    return {"imported": {}}


def save_tracking_data(data: dict[str, Any]) -> None:
    """Save the tracking file."""
    with open(TRACKING_FILE, "w") as f:
        json.dump(data, f, indent=2)


def format_issue_body(issue: dict[str, Any]) -> str:
    """Format the issue body with a link to the original."""
    original_url = issue["url"]
    created_at = datetime.fromisoformat(issue["createdAt"].replace("Z", "+00:00"))
    updated_at = datetime.fromisoformat(issue["updatedAt"].replace("Z", "+00:00"))

    body_parts = [
        f"**Original issue:** {original_url}",
        f"**Created:** {created_at.strftime('%Y-%m-%d')}",
        f"**Last updated:** {updated_at.strftime('%Y-%m-%d')}",
        "",
        "---",
        "",
        issue.get("body", "").strip() or "*No description provided.*",
    ]

    return "\n".join(body_parts)


def create_issue(issue: dict[str, Any]) -> str:
    """Create an issue in the fork repository."""
    upstream_number = issue["number"]
    title = issue["title"]
    body = format_issue_body(issue)

    # Collect labels
    labels = [UPSTREAM_LABEL]
    if issue.get("labels"):
        labels.extend([label["name"] for label in issue["labels"]])

    print(f"  Creating issue #{upstream_number}: {title[:60]}...")

    # Create the issue
    cmd = [
        "issue",
        "create",
        "--repo",
        FORK_REPO,
        "--title",
        title,
        "--body",
        body,
    ]

    # Add labels
    for label in labels:
        cmd.extend(["--label", label])

    output = run_gh_command(cmd)
    fork_issue_url = output.strip()

    print(f"  ✓ Created: {fork_issue_url}")
    return fork_issue_url


def main() -> None:
    """Main entry point."""
    print("=" * 70)
    print("Import Upstream Issues")
    print("=" * 70)
    print()

    # Ensure label exists
    ensure_label_exists()
    print()

    # Fetch upstream issues
    upstream_issues = fetch_upstream_issues()
    print()

    # Load tracking data
    tracking_data = load_tracking_data()
    imported = tracking_data.get("imported", {})

    # Filter out already imported issues
    to_import = [issue for issue in upstream_issues if str(issue["number"]) not in imported]

    if not to_import:
        print("✓ All issues have already been imported!")
        return

    print(f"Found {len(to_import)} issues to import")
    print(f"Already imported: {len(imported)}")
    print()

    # Import issues
    print("Starting import...")
    print()

    success_count = 0
    failed = []

    for idx, issue in enumerate(to_import, 1):
        upstream_number = issue["number"]
        try:
            print(f"[{idx}/{len(to_import)}]", end=" ")
            fork_url = create_issue(issue)

            # Track the import
            imported[str(upstream_number)] = {
                "fork_url": fork_url,
                "imported_at": datetime.now().isoformat(),
                "title": issue["title"],
            }

            # Save after each successful import
            tracking_data["imported"] = imported
            save_tracking_data(tracking_data)

            success_count += 1

            # Rate limiting
            if idx < len(to_import):
                time.sleep(RATE_LIMIT_DELAY)

        except Exception as e:
            print(f"  ✗ Failed to create issue #{upstream_number}: {e}")
            failed.append((upstream_number, str(e)))
            # Continue with next issue

    print()
    print("=" * 70)
    print("Import Complete")
    print("=" * 70)
    print(f"Successfully imported: {success_count}/{len(to_import)}")
    print(f"Total imported (all time): {len(imported)}")

    if failed:
        print(f"\nFailed issues ({len(failed)}):")
        for number, error in failed:
            print(f"  #{number}: {error}")

    print()
    print(f"Tracking data saved to: {TRACKING_FILE}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress has been saved.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}", file=sys.stderr)
        sys.exit(1)
