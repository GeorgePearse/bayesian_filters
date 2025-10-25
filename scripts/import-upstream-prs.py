#!/usr/bin/env python3
"""
Import open pull requests from the original rlabbe/filterpy repository.

This script attempts to automatically integrate upstream PRs by:
1. Fetching PR branches from upstream
2. Cherry-picking commits into isolated worktrees
3. Validating changes with pre-commit hooks and tests
4. Creating new PRs in the fork (draft or regular)

For PRs that integrate successfully, a new PR is created in the fork.
For PRs with conflicts or failures, a draft PR is created showing the error details.
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any
import textwrap
import shutil


UPSTREAM_REPO = "rlabbe/filterpy"
FORK_REPO = "GeorgePearse/bayesian_filters"
UPSTREAM_PR_LABEL = "from-upstream-pr"
UPSTREAM_PR_BLOCKED_LABEL = "upstream-pr-blocked"
UPSTREAM_PR_STALE_LABEL = "upstream-pr-stale"
TRACKING_FILE = "imported-prs.json"
WORKTREE_BASE = Path("/tmp/bayesian-filters-pr-worktrees")
BASE_BRANCH = "master"
RATE_LIMIT_DELAY = 1.5  # seconds between PR creation
STALE_DAYS = 365 * 2  # 2 years


@dataclass
class PRData:
    """Metadata for an upstream PR."""
    number: int
    title: str
    body: str
    url: str
    headRefName: str
    labels: list[dict[str, Any]] = field(default_factory=list)
    createdAt: str = ""
    updatedAt: str = ""


@dataclass
class IntegrationResult:
    """Result of attempting to integrate a PR."""
    success: bool
    has_conflicts: bool
    error_message: str | None = None
    commits_applied: int = 0
    conflict_files: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of validating integrated PR."""
    precommit_passed: bool = True
    tests_passed: bool = True
    precommit_output: str = ""
    test_output: str = ""


def find_git_root() -> Path:
    """Find the git repository root."""
    current = Path.cwd()
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    # If not found, return the parent of the scripts directory
    return Path(__file__).parent.parent


_GIT_ROOT = find_git_root()


def run_command(
    args: list[str], cwd: Path | None = None, check: bool = True
) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        # Default to git root if no cwd specified
        if cwd is None:
            cwd = _GIT_ROOT
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            cwd=cwd,
            check=False,
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)


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


def ensure_labels_exist() -> None:
    """Ensure all required labels exist in the fork."""
    labels = [
        (UPSTREAM_PR_LABEL, "0366d6", "PR imported from upstream rlabbe/filterpy repository"),
        (UPSTREAM_PR_BLOCKED_LABEL, "d73a4a", "Upstream PR blocked by conflicts or errors"),
        (UPSTREAM_PR_STALE_LABEL, "cccccc", "Upstream PR is stale (2+ years old)"),
        ("needs-precommit-fix", "ffd700", "Pre-commit hooks failed after integration"),
        ("failing-tests", "ff6b6b", "Tests failed after integration"),
        ("low-risk", "a2eeef", "Low-risk PR (docs, typos, etc)"),
    ]

    for label_name, color, description in labels:
        try:
            run_gh_command(
                [
                    "label",
                    "create",
                    label_name,
                    "--repo",
                    FORK_REPO,
                    "--description",
                    description,
                    "--color",
                    color,
                ]
            )
            print(f"  Created label: {label_name}")
        except subprocess.CalledProcessError:
            # Label might already exist
            pass


def ensure_upstream_remote() -> None:
    """Ensure upstream remote is configured."""
    code, stdout, _ = run_command(["git", "remote"])

    if "upstream" not in stdout:
        print(f"Adding upstream remote: {UPSTREAM_REPO}")
        run_command([
            "git",
            "remote",
            "add",
            "upstream",
            f"https://github.com/{UPSTREAM_REPO}.git",
        ])

    print("Fetching from upstream...")
    run_command(["git", "fetch", "upstream", "master"])


def fetch_upstream_prs() -> list[PRData]:
    """Fetch all open PRs from upstream."""
    print(f"Fetching open PRs from {UPSTREAM_REPO}...")
    output = run_gh_command(
        [
            "pr",
            "list",
            "--repo",
            UPSTREAM_REPO,
            "--state",
            "open",
            "--limit",
            "100",
            "--json",
            "number,title,body,labels,createdAt,updatedAt,url,headRefName",
        ]
    )

    prs_data = json.loads(output)
    prs = [PRData(**pr) for pr in prs_data]
    print(f"Found {len(prs)} open PRs")
    return prs


def load_tracking_data() -> dict[str, Any]:
    """Load the tracking file."""
    if Path(TRACKING_FILE).exists():
        with open(TRACKING_FILE) as f:
            return json.load(f)
    return {"imported": {}}


def save_tracking_data(data: dict[str, Any]) -> None:
    """Save the tracking file."""
    with open(TRACKING_FILE, "w") as f:
        json.dump(data, f, indent=2)


def is_pr_stale(pr: PRData) -> bool:
    """Check if PR is older than STALE_DAYS."""
    created = datetime.fromisoformat(pr.createdAt.replace("Z", "+00:00"))
    age_days = (datetime.now(created.tzinfo) - created).days
    return age_days > STALE_DAYS


def create_worktree(pr: PRData) -> Path | None:
    """Create an isolated git worktree for the PR."""
    branch_name = f"upstream-pr-{pr.number}-{pr.title[:30].lower().replace(' ', '-').replace('/', '-')}"
    branch_name = "".join(c for c in branch_name if c.isalnum() or c == "-")

    worktree_path = WORKTREE_BASE / f"pr-{pr.number}"

    # Clean up if it exists
    if worktree_path.exists():
        run_command(["git", "worktree", "remove", str(worktree_path), "--force"])

    WORKTREE_BASE.mkdir(parents=True, exist_ok=True)

    code, _, stderr = run_command([
        "git",
        "worktree",
        "add",
        str(worktree_path),
        "-b",
        branch_name,
        f"upstream/{BASE_BRANCH}",
    ])

    if code != 0:
        print(f"  Failed to create worktree: {stderr}")
        return None

    return worktree_path


def get_pr_commits(pr: PRData) -> list[str]:
    """Get commit SHAs from the PR."""
    try:
        output = run_gh_command([
            "pr",
            "view",
            str(pr.number),
            "--repo",
            UPSTREAM_REPO,
            "--json",
            "commits",
        ])
        commits_data = json.loads(output)
        return [c["oid"] for c in commits_data.get("commits", [])]
    except Exception as e:
        print(f"  Failed to get commits: {e}")
        return []


def integrate_pr_commits(worktree_path: Path, pr: PRData) -> IntegrationResult:
    """Attempt to integrate PR commits."""
    commits = get_pr_commits(pr)

    if not commits:
        return IntegrationResult(
            success=False,
            has_conflicts=False,
            error_message="Could not fetch commits from upstream PR",
        )

    applied = 0
    conflict_files = []

    for i, commit_sha in enumerate(commits):
        code, _, stderr = run_command(
            ["git", "cherry-pick", commit_sha],
            cwd=worktree_path,
        )

        if code != 0:
            # Check if it's a conflict
            if "conflict" in stderr.lower():
                # Get conflicted files
                code2, stdout2, _ = run_command(
                    ["git", "diff", "--name-only", "--diff-filter=U"],
                    cwd=worktree_path,
                )
                if code2 == 0:
                    conflict_files = stdout2.strip().split("\n")

                # Abort the cherry-pick
                run_command(["git", "cherry-pick", "--abort"], cwd=worktree_path)

                return IntegrationResult(
                    success=False,
                    has_conflicts=True,
                    error_message=f"Conflict on commit {i+1}/{len(commits)}",
                    commits_applied=applied,
                    conflict_files=conflict_files,
                )
            else:
                # Other error
                return IntegrationResult(
                    success=False,
                    has_conflicts=False,
                    error_message=f"Cherry-pick failed: {stderr[:200]}",
                    commits_applied=applied,
                )

        applied += 1

    return IntegrationResult(
        success=True,
        has_conflicts=False,
        commits_applied=applied,
    )


def run_validation(worktree_path: Path, skip_precommit: bool = False, skip_tests: bool = False) -> ValidationResult:
    """Run validation checks on integrated PR."""
    result = ValidationResult()

    if not skip_precommit:
        print(f"    Running pre-commit hooks...")
        code, stdout, stderr = run_command(
            ["pre-commit", "run", "--all-files"],
            cwd=worktree_path,
        )
        result.precommit_passed = code == 0
        result.precommit_output = stderr or stdout
        if code != 0:
            print(f"    Pre-commit failed: {stderr[:100]}")

    if not skip_tests:
        print(f"    Running tests...")
        code, stdout, stderr = run_command(
            ["pytest", "-x"],
            cwd=worktree_path,
        )
        result.tests_passed = code == 0
        result.test_output = stderr or stdout
        if code != 0:
            print(f"    Tests failed: {stderr[:100]}")

    return result


def get_branch_name(pr: PRData) -> str:
    """Generate branch name for PR."""
    branch_name = f"upstream-pr-{pr.number}-{pr.title[:30].lower().replace(' ', '-').replace('/', '-')}"
    branch_name = "".join(c for c in branch_name if c.isalnum() or c == "-")
    return branch_name


def format_pr_body(pr: PRData, validation: ValidationResult) -> str:
    """Format body for new PR in fork."""
    body_parts = [
        f"**Upstream PR:** {pr.url}",
        f"**Author:** Imported from upstream",
        "",
        "---",
        "",
        "### Original Description",
        "",
        pr.body or "*No description provided*",
        "",
        "---",
        "",
        "### Integration Notes",
        "",
    ]

    if not validation.precommit_passed:
        body_parts.append("- Pre-commit hooks failed - may need fixes")

    if not validation.tests_passed:
        body_parts.append("- Tests failed - may need investigation")

    if validation.precommit_passed and validation.tests_passed:
        body_parts.append("- Validation passed")

    return "\n".join(body_parts)


def format_draft_pr_body(pr: PRData, integration: IntegrationResult) -> str:
    """Format body for draft PR when integration fails."""
    body_parts = [
        f"**Upstream PR:** {pr.url}",
        "",
        "---",
        "",
        "### Status: Integration Failed",
        "",
    ]

    if integration.has_conflicts:
        body_parts.extend([
            f"⚠️ **Conflicts detected** on commit {integration.commits_applied + 1}/{integration.commits_applied or 1}",
            "",
            "**Conflicted files:**",
            "",
        ])
        for f in integration.conflict_files:
            body_parts.append(f"- `{f}`")
    else:
        body_parts.append(f"❌ **Error:** {integration.error_message}")

    body_parts.extend([
        "",
        "---",
        "",
        "### Original Description",
        "",
        pr.body or "*No description provided*",
    ])

    return "\n".join(body_parts)


def create_pr_in_fork(
    pr: PRData,
    branch_name: str,
    validation: ValidationResult,
    draft: bool = False,
) -> str | None:
    """Create a PR in the fork."""
    pr_type = "draft PR" if draft else "PR"
    print(f"    Creating {pr_type} in fork...")

    body = format_pr_body(pr, validation)

    labels = [UPSTREAM_PR_LABEL]

    # Add conditional labels
    if not validation.precommit_passed:
        labels.append("needs-precommit-fix")
    if not validation.tests_passed:
        labels.append("failing-tests")
    if is_pr_stale(pr):
        labels.append(UPSTREAM_PR_STALE_LABEL)

    # Check if it's a docs-only or low-risk change
    if _is_low_risk_pr(pr):
        labels.append("low-risk")

    try:
        cmd = [
            "pr",
            "create",
            "--repo",
            FORK_REPO,
            "--base",
            BASE_BRANCH,
            "--head",
            branch_name,
            "--title",
            f"[upstream PR #{pr.number}] {pr.title}",
            "--body",
            body,
        ]

        if draft:
            cmd.append("--draft")

        for label in labels:
            cmd.extend(["--label", label])

        output = run_gh_command(cmd)
        pr_url = output.strip()
        print(f"    Created {pr_type}: {pr_url}")
        return pr_url
    except Exception as e:
        print(f"    Failed to create {pr_type}: {e}")
        return None




def push_branch(worktree_path: Path, branch_name: str) -> bool:
    """Push branch to fork."""
    print(f"    Pushing branch: {branch_name}")
    code, _, stderr = run_command(
        ["git", "push", "-u", "origin", branch_name],
        cwd=worktree_path,
    )

    if code != 0:
        print(f"    Failed to push: {stderr[:100]}")
        return False

    return True


def cleanup_worktree(worktree_path: Path) -> None:
    """Remove the worktree."""
    if worktree_path.exists():
        run_command(["git", "worktree", "remove", str(worktree_path), "--force"])


def _is_low_risk_pr(pr: PRData) -> bool:
    """Check if PR is low-risk (docs, typos, etc)."""
    title_lower = pr.title.lower()
    body_lower = pr.body.lower() if pr.body else ""

    keywords = ["typo", "doc", "readme", "comment", "fix", "update", "remove"]
    return any(kw in title_lower or kw in body_lower for kw in keywords)


def process_pr(
    pr: PRData,
    tracking_data: dict[str, Any],
    dry_run: bool = False,
    skip_precommit: bool = False,
    skip_tests: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """Process a single PR."""
    pr_str = f"[{pr.number}] {pr.title[:60]}"
    print(f"\n  Processing PR #{pr.number}: {pr.title[:60]}")

    # Create worktree
    worktree_path = create_worktree(pr)
    if not worktree_path:
        print(f"    Failed to create worktree")
        return False, {}

    print(f"    Created worktree: {worktree_path}")

    try:
        # Attempt integration
        print(f"    Cherry-picking commits...")
        integration = integrate_pr_commits(worktree_path, pr)
        branch_name = get_branch_name(pr)

        # Push branch regardless of integration success
        if not push_branch(worktree_path, branch_name):
            print(f"    Failed to push branch")
            return False, {}

        if not integration.success:
            print(f"    Integration failed: {integration.error_message}")

            # Create draft PR with failure details
            if not dry_run:
                # Create a minimal validation result for draft PR
                draft_validation = ValidationResult(
                    precommit_passed=False,
                    tests_passed=False,
                    precommit_output="Integration failed - draft PR created",
                    test_output=integration.error_message or "",
                )
                # Use custom body for failed integrations
                body = format_draft_pr_body(pr, integration)

                labels = [UPSTREAM_PR_LABEL, UPSTREAM_PR_BLOCKED_LABEL]
                if is_pr_stale(pr):
                    labels.append(UPSTREAM_PR_STALE_LABEL)

                try:
                    cmd = [
                        "pr",
                        "create",
                        "--repo",
                        FORK_REPO,
                        "--base",
                        BASE_BRANCH,
                        "--head",
                        branch_name,
                        "--title",
                        f"[upstream PR #{pr.number}] {pr.title}",
                        "--body",
                        body,
                        "--draft",
                    ]

                    for label in labels:
                        cmd.extend(["--label", label])

                    output = run_gh_command(cmd)
                    pr_url = output.strip()
                    print(f"    Created draft PR: {pr_url}")
                except Exception as e:
                    print(f"    Failed to create draft PR: {e}")
                    pr_url = None
            else:
                pr_url = "[DRY RUN] would create draft PR"

            return False, {
                "status": "draft_pr_created",
                "fork_pr_url": pr_url,
                "upstream_pr_url": pr.url,
                "attempt_date": datetime.now().isoformat(),
                "conflict_files": integration.conflict_files,
                "error": integration.error_message,
                "branch_name": branch_name,
                "is_draft": True,
            }

        print(f"    Successfully cherry-picked {integration.commits_applied} commit(s)")

        # Validate
        validation = run_validation(worktree_path, skip_precommit, skip_tests)

        # Create PR
        if not dry_run:
            pr_url = create_pr_in_fork(pr, branch_name, validation, draft=False)
        else:
            pr_url = "[DRY RUN] would create PR"

        if not pr_url:
            print(f"    Failed to create PR")
            return False, {}

        print(f"    Success!")

        return True, {
            "status": "pr_created",
            "fork_pr_url": pr_url,
            "upstream_pr_url": pr.url,
            "integration_date": datetime.now().isoformat(),
            "had_conflicts": False,
            "validation_passed": validation.precommit_passed and validation.tests_passed,
            "branch_name": branch_name,
            "is_draft": False,
        }

    finally:
        # Always cleanup
        cleanup_worktree(worktree_path)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Import upstream pull requests from rlabbe/filterpy"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without creating PRs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from tracking file (skip already-processed PRs)",
    )
    parser.add_argument(
        "--pr",
        type=int,
        help="Process only specific PR number",
    )
    parser.add_argument(
        "--skip-precommit",
        action="store_true",
        help="Skip pre-commit hook validation",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest validation",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Import Upstream Pull Requests")
    print("=" * 70)
    print()

    # Setup
    print("Setting up labels...")
    ensure_labels_exist()
    print()

    print("Configuring upstream remote...")
    ensure_upstream_remote()
    print()

    # Fetch
    upstream_prs = fetch_upstream_prs()
    print()

    # Filter
    if args.pr:
        upstream_prs = [p for p in upstream_prs if p.number == args.pr]
        if not upstream_prs:
            print(f"PR #{args.pr} not found")
            return

    # Load tracking
    tracking_data = load_tracking_data()
    imported = tracking_data.get("imported", {})

    if args.resume:
        to_process = [p for p in upstream_prs if str(p.number) not in imported]
        print(f"Found {len(upstream_prs)} open PRs")
        print(f"Already processed: {len(imported)}")
        print(f"To process: {len(to_process)}")
    else:
        to_process = upstream_prs
        print(f"Found {len(upstream_prs)} open PRs")
        if args.resume is False and imported:
            print(f"(Already processed: {len(imported)})")
    print()

    if not to_process:
        print("All PRs have been processed!")
        return

    if args.dry_run:
        print("[DRY RUN MODE] - No changes will be made")
        print()

    # Process PRs
    print("=" * 70)
    print("Processing PRs")
    print("=" * 70)
    print()

    success_count = 0
    failed_count = 0

    for idx, pr in enumerate(to_process, 1):
        try:
            success, pr_data = process_pr(
                pr,
                tracking_data,
                dry_run=args.dry_run,
                skip_precommit=args.skip_precommit,
                skip_tests=args.skip_tests,
            )

            if success:
                success_count += 1
                imported[str(pr.number)] = pr_data
            else:
                failed_count += 1
                if pr_data:
                    imported[str(pr.number)] = pr_data

            # Save progress
            tracking_data["imported"] = imported
            save_tracking_data(tracking_data)

            # Rate limiting
            if idx < len(to_process):
                time.sleep(RATE_LIMIT_DELAY)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            save_tracking_data(tracking_data)
            sys.exit(1)
        except Exception as e:
            print(f"  Unexpected error: {e}")
            failed_count += 1

    # Summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Successfully integrated: {success_count}/{len(to_process)}")
    print(f"Blocked/failed: {failed_count}/{len(to_process)}")
    print(f"Total imported (all time): {len(imported)}")
    print()
    print(f"Tracking data saved to: {TRACKING_FILE}")
    print()

    if success_count > 0 or failed_count > 0:
        print("Next steps:")
        print(f"  Review PRs: https://github.com/{FORK_REPO}/pulls?q=label:{UPSTREAM_PR_LABEL}")
        if failed_count > 0:
            print(f"  Review draft PRs (needs fixes): https://github.com/{FORK_REPO}/pulls?q=label:{UPSTREAM_PR_BLOCKED_LABEL}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}", file=sys.stderr)
        sys.exit(1)
