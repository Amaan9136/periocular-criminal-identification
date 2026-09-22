# Claude Code instructions for this repo

This file governs how Claude Code should review and act on pull requests in
**periocular-criminal-identification**, including the batch of auto-generated
PRs currently open. Follow it every time, not just once.

## Project context (read before reviewing anything)

This is a periocular (eye-region) biometric identification tool for
investigative lead generation. It is explicitly **not** a courtroom-grade
identification system, and every match it produces must remain a
similarity-ranked candidate for human review, never an automated
accusation. See `README.md` (Honest Limitations & Responsible Use) and
`CONTRIBUTING.md` before touching anything.

## How to review each PR

For every open PR:

1. `gh pr diff <number>` to read the full change.
2. `gh pr checks <number>` to see CI status. Treat failing checks as a hard
   block on auto-merge regardless of what the diff looks like.
3. Run the app locally / run any existing tests against the branch if
   feasible, and note what you actually verified in your summary.
4. Classify the PR into one of: **auto-mergeable**, **needs my sign-off**,
   or **reject with comment** (see rules below).
5. Leave a comment on the PR explaining the classification and reasoning,
   even for ones you merge.

## Hard blockers — never auto-merge, always flag for my sign-off

A PR goes to "needs my sign-off" (never auto-merged, no exceptions) if it:

- Adds, changes, or hardcodes any **model download URL** as a default,
  especially anything touching `services/model_registry.py`,
  `GPEN_DOWNLOAD_URL_ENV`, or the models install/download flow. Any
  download source must remain something the user supplies at install time
  (form field or their own env var) — never a baked-in default in source,
  settings, or a database.
- Adds a new model, especially anything resembling a face-swap or identity
  synthesis model (e.g. inswapper-style). Per `CONTRIBUTING.md`, anything
  that could fabricate a photorealistic but fake identity is out of scope
  and should be flagged, not merged.
- Changes how results are presented in a way that makes a match look like
  an automated identification instead of a candidate for human review.
- Touches biometric data handling, embedding storage, or anything under
  `services/sqlite_store.py` / `services/vector_store.py` in a way that
  could affect what's persisted or how it's exposed.
- Includes real biometric data (real people's photos, embeddings, or
  `data/*.db` files) in the diff — reject with a comment, don't just flag.
- Silently changes behavior of `reconstruction_mode='hallucinate'` or the
  `inswapper_128` exclusion to make either look functional when it isn't.
- Modifies `CLAUDE.md` itself, `CONTRIBUTING.md`, or `CODE_OF_CONDUCT.md`.

## Auto-mergeable if all of these hold

A PR may be merged without waiting for me only if:

- CI passes.
- It's a single, focused change (docs fix, dependency bump with no breaking
  changes, test addition, lint/formatting, small bugfix) matching the
  "keep PRs focused" norm in `CONTRIBUTING.md`.
- It does not touch any of the hard-blocker areas above.
- It doesn't change public API shapes (`schemas.py`, route signatures) in
  a breaking way.
- The PR description or diff makes the intent clear enough that you don't
  need to guess.

## Everything else

Default to "needs my sign-off" when uncertain. Summarize the diff, what you
checked, and your recommendation, but wait for explicit approval before
merging. Do not merge multiple PRs that touch overlapping files without
checking for conflicts between them first — resolve ordering, don't merge
blind.

## Style / conventions to enforce while reviewing

- No code comments unless the existing file already uses them consistently.
- Match existing formatting exactly (this codebase uses compact,
  low-whitespace Python and single-line JS function bodies in
  `static/js/app.js`) — don't let a PR reformat untouched code.
- Commit messages should follow whatever convention is in
  `.github/COMMIT_CONVENTION.md` if present.

## After acting on all PRs

Post one final summary comment (or write it to me directly) listing: what
was merged, what's waiting on my approval and why, and what was rejected
and why.