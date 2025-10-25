# Importing Issues from Upstream

This document explains how to import open issues from the original `rlabbe/filterpy` repository into this fork.

## Overview

The `import-upstream-issues.py` script automatically imports all open issues from the upstream repository, creating corresponding issues in this fork with links back to the originals.

## Features

- ✅ Imports all open issues from `rlabbe/filterpy`
- ✅ Adds link to original issue in the body
- ✅ Includes original creation and update dates
- ✅ Automatically adds `from-upstream` label
- ✅ Preserves original labels (if any)
- ✅ Tracks imported issues to avoid duplicates
- ✅ Can be safely re-run (idempotent)
- ✅ Handles rate limiting automatically

## Prerequisites

1. **GitHub CLI installed**: The script uses `gh` command
   ```bash
   # Check if installed
   gh --version

   # Install on macOS
   brew install gh

   # Install on Linux
   sudo apt install gh
   # or
   sudo dnf install gh
   ```

2. **Authenticated with GitHub**:
   ```bash
   gh auth login
   ```

3. **Repository access**: You need write access to the fork repository

## Usage

### Basic Usage

Simply run the script:

```bash
./import-upstream-issues.py
```

Or with Python explicitly:

```bash
python3 import-upstream-issues.py
```

### What the Script Does

1. **Creates the `from-upstream` label** (if it doesn't exist)
   - Color: Blue (#0366d6)
   - Description: "Issue imported from upstream rlabbe/filterpy repository"

2. **Fetches all open issues** from `rlabbe/filterpy`

3. **Creates corresponding issues** in `GeorgePearse/filterpy` with:
   - Same title as original
   - Modified body that includes:
     ```markdown
     **Original issue:** https://github.com/rlabbe/filterpy/issues/123
     **Created:** 2023-01-15
     **Last updated:** 2024-03-20

     ---

     [Original issue body content]
     ```
   - Labels: `from-upstream` + any original labels

4. **Tracks progress** in `imported-issues.json`
   - Records which issues have been imported
   - Can safely re-run without creating duplicates

5. **Handles errors gracefully**
   - Continues on failure
   - Reports failed issues at the end
   - Saves progress after each successful import

### Example Output

```
======================================================================
Import Upstream Issues
======================================================================

Ensuring 'from-upstream' label exists...
✓ Created 'from-upstream' label

Fetching open issues from rlabbe/filterpy...
✓ Found 71 open issues

Found 71 issues to import
Already imported: 0

Starting import...

[1/71]   Creating issue #319: Circular import of filterpy itself prevents the use of pip...
  ✓ Created: https://github.com/GeorgePearse/filterpy/issues/1
[2/71]   Creating issue #318: control  (Bu) in UKF...
  ✓ Created: https://github.com/GeorgePearse/filterpy/issues/2
...

======================================================================
Import Complete
======================================================================
Successfully imported: 71/71
Total imported (all time): 71

Tracking data saved to: imported-issues.json
```

## Tracking File

The script creates `imported-issues.json` to track what's been imported:

```json
{
  "imported": {
    "319": {
      "fork_url": "https://github.com/GeorgePearse/filterpy/issues/1",
      "imported_at": "2025-10-20T16:30:00.123456",
      "title": "Circular import of filterpy itself prevents the use of pip"
    },
    "318": {
      "fork_url": "https://github.com/GeorgePearse/filterpy/issues/2",
      "imported_at": "2025-10-20T16:30:02.234567",
      "title": "control  (Bu) in UKF"
    }
  }
}
```

**Note**: This file should be added to `.gitignore` if you don't want to track it in version control.

## Re-running the Script

The script is **idempotent** - you can run it multiple times:

- ✅ Already imported issues are skipped
- ✅ Only new upstream issues are imported
- ✅ No duplicates will be created

This is useful when:
- New issues are opened in the upstream repository
- The script was interrupted
- Previous run had failures

## Rate Limiting

The script includes automatic rate limiting:
- 1.5 second delay between issue creation
- Prevents hitting GitHub API limits
- ~40 issues per minute

For 71 issues, expect the script to take approximately 2-3 minutes.

## Troubleshooting

### "Error: Could not resolve to a Repository"

Make sure you have access to the repository and are authenticated:
```bash
gh auth status
gh auth login
```

### "Label already exists"

This is expected behavior. The script will use the existing label.

### Script interrupted

No problem! The script saves progress after each issue. Just run it again:
```bash
./import-upstream-issues.py
```

### Want to start fresh?

Delete the tracking file and run again:
```bash
rm imported-issues.json
./import-upstream-issues.py
```

**Warning**: This will create duplicate issues if you've already imported some.

## Customization

You can edit the script to customize:

- `UPSTREAM_LABEL`: The label name (default: `from-upstream`)
- `RATE_LIMIT_DELAY`: Seconds between issue creation (default: 1.5)
- `format_issue_body()`: How the issue body is formatted

## Maintenance

### Updating Imported Issues

The script only imports once. To update an imported issue with changes from upstream:
- Manually copy updates from the original issue
- Or close and re-import (requires manual deletion and tracking file edit)

### Closing Resolved Upstream Issues

If an upstream issue is closed:
- The script won't import it (only imports open issues)
- Existing imported issues in your fork won't be automatically closed
- Consider manually closing them or adding automation

## Statistics

As of the last check:
- **Total upstream issues**: 233 (all time)
- **Open upstream issues**: 71
- **Issues this script will import**: 71

## See Also

- [Original repository](https://github.com/rlabbe/filterpy)
- [Original issues](https://github.com/rlabbe/filterpy/issues)
- [GitHub CLI documentation](https://cli.github.com/manual/)
