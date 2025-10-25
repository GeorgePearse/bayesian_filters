# Agent Instructions

## ⚠️ CRITICAL: Pull Request Guidelines

### DO NOT SUBMIT PRs TO FILTERPY

**🚫 ABSOLUTELY NO PULL REQUESTS TO:** https://github.com/rlabbe/filterpy

This is a **STANDALONE FORK** for independent development.

- **ALL PRs MUST target:** https://github.com/GeorgePearse/bayesian_filters (THIS REPOSITORY ONLY)
- **NEVER** submit changes to the original filterpy repository
- **NEVER** attempt to upstream changes
- **NEVER** contribute to https://github.com/rlabbe/filterpy under any circumstances

### Why?

This is a personal/independent fork with its own roadmap, features, and direction. All development work, features, fixes, and improvements are **exclusively for this fork**. There is no intention to contribute back to the original project.

### Pull Request Checklist

Before creating ANY pull request, verify:
- ✅ Target is: `GeorgePearse/bayesian_filters` (THIS REPO)
- ✅ NOT targeting: `rlabbe/filterpy` (FORBIDDEN)
- ✅ Base branch is set correctly for this fork
- ✅ You are NOT attempting to contribute upstream

## 📝 Agent Documentation Guidelines

### Where to Write Documentation

When agents need to create documentation files (design docs, analysis reports, guides, etc.), they should **write to the `scratch_files/` directory** instead of the repository root.

**Example:**
- ❌ DON'T: Create `MY_ANALYSIS.md` in root
- ✅ DO: Create `scratch_files/MY_ANALYSIS.md`

### Why?

The repository root should contain only essential documentation files:
- `README.md` - Project overview
- `AGENTS.md` - Agent guidelines (this file)
- Other critical docs

The `scratch_files/` directory is where agents can freely write:
- Analysis and investigation reports
- Design documents
- Planning notes
- Implementation guides
- Troubleshooting docs
- Architecture diagrams
- Any other supporting documentation

### File Naming

Agent-generated documentation in `scratch_files/` can use any naming convention:
- Capitalized markdown files are fine (e.g., `ANALYSIS.md`, `DESIGN.md`)
- Descriptive names are encouraged
- Dates/timestamps are helpful (e.g., `2025-10-25_investigation.md`)
