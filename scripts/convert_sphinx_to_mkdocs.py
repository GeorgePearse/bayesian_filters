#!/usr/bin/env python3
"""
Convert Sphinx RST documentation to MkDocs Markdown.

This script converts the existing Sphinx-based documentation (in docs/)
to MkDocs-compatible Markdown files (in docs-mkdocs/).
"""

import os
import re
import shutil
from pathlib import Path


def rst_to_markdown(content: str) -> str:
    """Convert RST content to Markdown."""

    # Headers - convert RST underline style to Markdown #
    # Level 1: ***
    content = re.sub(r"^(.+)\n\*+$", r"# \1", content, flags=re.MULTILINE)
    # Level 2: ===
    content = re.sub(r"^(.+)\n=+$", r"## \1", content, flags=re.MULTILINE)
    # Level 3: ---
    content = re.sub(r"^(.+)\n-+$", r"### \1", content, flags=re.MULTILINE)
    # Level 4: ^^^
    content = re.sub(r"^(.+)\n\^+$", r"#### \1", content, flags=re.MULTILINE)

    # Code blocks
    content = re.sub(
        r"\.\. code-block:: (\w+)\n\n((?:    .+\n)+)",
        lambda m: f"```{m.group(1)}\n{m.group(2).replace('    ', '')}\n```\n",
        content,
    )

    # Generic code blocks without language
    content = re.sub(r"::\n\n((?:    .+\n)+)", lambda m: f"```\n{m.group(1).replace('    ', '')}\n```\n", content)

    # Inline code
    content = re.sub(r"``([^`]+)``", r"`\1`", content)

    # Links: `text <url>`_
    content = re.sub(r"`([^<]+)<([^>]+)>`_", r"[\1](\2)", content)

    # Images
    content = re.sub(
        r"\.\. image:: (.+)\n(?:    :target: (.+)\n)?",
        lambda m: f"[![]({m.group(1)})]({m.group(2)})" if m.group(2) else f"![]({m.group(1)})",
        content,
    )

    # Remove toctree directives (handle separately in navigation)
    content = re.sub(r"\.\. toctree::.+?(?=\n\S|\Z)", "", content, flags=re.DOTALL)

    # Remove automodule/autoclass directives - mkdocstrings will handle these
    content = re.sub(r"\.\. auto(module|class|function):: (.+)", r"::: \2", content)

    # Citations/references [1]_
    content = re.sub(r"\[(\d+)\]_", r"[\1]", content)

    # Bold
    content = re.sub(r"\*\*(.+?)\*\*", r"**\1**", content)

    # Italic
    content = re.sub(r"\*([^\*]+)\*", r"*\1*", content)

    # Remove multiple blank lines
    content = re.sub(r"\n{3,}", "\n\n", content)

    return content.strip()


def convert_rst_file(rst_path: Path, md_path: Path):
    """Convert a single RST file to Markdown."""
    print(f"Converting {rst_path} -> {md_path}")

    with open(rst_path, "r", encoding="utf-8") as f:
        content = f.read()

    markdown = rst_to_markdown(content)

    # Ensure parent directory exists
    md_path.parent.mkdir(parents=True, exist_ok=True)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)


def main():
    """Main conversion function."""
    docs_dir = Path("docs")
    mkdocs_dir = Path("docs-mkdocs")

    # Create MkDocs directories
    (mkdocs_dir / "api").mkdir(parents=True, exist_ok=True)
    (mkdocs_dir / "filters").mkdir(parents=True, exist_ok=True)
    (mkdocs_dir / "algorithms").mkdir(parents=True, exist_ok=True)

    # Map Sphinx structure to MkDocs structure
    conversions = [
        # Kalman filters
        ("kalman/KalmanFilter.rst", "filters/kalman-filter.md"),
        ("kalman/ExtendedKalmanFilter.rst", "filters/extended-kalman-filter.md"),
        ("kalman/UnscentedKalmanFilter.rst", "filters/unscented-kalman-filter.md"),
        ("kalman/EnsembleKalmanFilter.rst", "filters/ensemble-kalman-filter.md"),
        ("kalman/InformationFilter.rst", "filters/information-filter.md"),
        ("kalman/SquareRootFilter.rst", "filters/square-root-filter.md"),
        ("kalman/FadingKalmanFilter.rst", "filters/fading-kalman-filter.md"),
        ("kalman/IMMEstimator.rst", "filters/imm-estimator.md"),
        ("kalman/MMAEFilterBank.rst", "filters/mmae-filter-bank.md"),
        # Other algorithms
        ("gh/GHFilter.rst", "algorithms/gh-filter.md"),
        ("gh/GHKFilter.rst", "algorithms/ghk-filter.md"),
        ("discrete_bayes/discrete_bayes.rst", "algorithms/discrete-bayes.md"),
        ("leastsq/leastsq.rst", "algorithms/least-squares.md"),
        ("monte_carlo/monte_carlo.rst", "algorithms/monte-carlo.md"),
        ("hinfinity/HInfinityFilter.rst", "filters/h-infinity-filter.md"),
        # API docs
        ("common/common.rst", "api/common.md"),
        ("stats/stats.rst", "api/stats.md"),
    ]

    for src, dst in conversions:
        src_path = docs_dir / src
        dst_path = mkdocs_dir / dst

        if src_path.exists():
            convert_rst_file(src_path, dst_path)
        else:
            print(f"Warning: {src_path} does not exist, skipping")

    # Convert index.rst to index.md
    index_src = docs_dir / "index.rst"
    if index_src.exists():
        # The index already exists, so let's just add the main content
        print(f"Note: {mkdocs_dir / 'index.md'} already exists, not overwriting")

    print("\nConversion complete!")
    print(f"Converted files are in {mkdocs_dir}/")
    print("\nNext steps:")
    print("1. Review converted files for any formatting issues")
    print("2. Test with 'mkdocs serve'")
    print("3. Adjust mkdocs.yml navigation if needed")


if __name__ == "__main__":
    main()
