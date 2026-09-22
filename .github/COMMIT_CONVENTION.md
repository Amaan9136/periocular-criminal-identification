# Commit Message Convention

This repo follows **[Conventional Commits](https://www.conventionalcommits.org/)**. It keeps history readable, enables auto-generated changelogs, and makes `git log --oneline` actually useful.

## Format

```
<type>(<optional scope>): <short description>

[optional body]

[optional footer(s)]
```

## Type tags

| Tag | Use for |
|---|---|
| `feat` | A new feature (e.g. new endpoint, new dashboard page) |
| `fix` | A bug fix |
| `docs` | Documentation only changes (README, comments, this file) |
| `style` | Formatting, whitespace, missing semicolons — no logic change |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | A change that improves performance (e.g. faster similarity search) |
| `test` | Adding or correcting tests |
| `build` | Changes to build system or dependencies (`requirements.txt`, etc.) |
| `ci` | Changes to CI configuration/workflows |
| `chore` | Maintenance tasks that don't modify src or test files |
| `revert` | Reverts a previous commit |
| `security` | A fix specifically addressing a security or privacy issue |

## Scope (optional but encouraged)

Use the affected area, e.g. `api`, `dashboard`, `insightface`, `sqlite`, `models`, `docs`, `config`.

## Examples

```
feat(api): add pagination to GET /api/models

fix(sqlite): prevent WAL file growth on repeated small inserts

docs(readme): add architecture diagram and AI/ML model table

refactor(insightface): extract periocular crop math into a helper

perf(vector-store): use argpartition instead of full sort for top-k

security(dashboard): strip EXIF metadata before saving uploaded probe images

chore: bump onnxruntime to 1.26.0
```

## Breaking changes

Add `!` after the type/scope and a `BREAKING CHANGE:` footer:

```
feat(api)!: change /search response shape to include per-image warnings

BREAKING CHANGE: `SearchResponse.warnings` is now required and always
present (previously omitted when empty).
```

## Pull request titles

PR titles should follow the same convention — they're often used as the squash-merge commit message.
