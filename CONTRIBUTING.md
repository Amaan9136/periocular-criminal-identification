# Contributing to Periocular Criminal Identification

Thanks for considering a contribution — this project is open source specifically so other people working on biometric research, forensic tooling, or FastAPI/ML systems can build on it. This guide covers how to get set up and how changes get merged.

## Code of Conduct

By participating, you agree to uphold our [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- 🐛 **Report bugs** via [GitHub Issues](https://github.com/Amaan9136/periocular-criminal-identification/issues)
- 💡 **Suggest features or improvements** (open an issue first for anything non-trivial, so we can discuss the approach)
- 📝 **Improve documentation** — README clarity, code comments, examples
- 🧪 **Add tests** — the project currently has no automated test suite; this is high-value
- 🔬 **Fairness/bias evaluation** — if you can help benchmark the pipeline across demographic subgroups, that's especially welcome given the sensitivity of this application
- 🧹 **Refactor or optimize** — e.g. swapping the exact cosine scan for an ANN index (FAISS/HNSW) for large galleries

## Getting set up

```bash
git clone https://github.com/Amaan9136/periocular-criminal-identification.git
cd periocular-criminal-identification
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
```

## Development workflow

1. **Fork** the repo and create a branch off `main`:
   ```bash
   git checkout -b feat/short-description
   ```
2. Make your changes. Keep pull requests focused — one logical change per PR is much easier to review than a bundle of unrelated fixes.
3. If you're changing behavior (not just docs/refactors), please test it manually against the dashboard or `/docs` Swagger UI, and note how you tested it in the PR description.
4. Commit using the project's [commit message convention](.github/COMMIT_CONVENTION.md).
5. Push and open a **Pull Request** against `main`, describing:
   - What the change does and why
   - Any trade-offs or limitations it introduces
   - How you tested it

## Guidelines specific to this project

Because this is a biometric identification tool, a few extra norms apply:

- **Don't silently paper over limitations.** If a model or mode doesn't work as advertised (see the `hallucinate` mode and the excluded `inswapper_128` model in the README for precedent), document that honestly rather than making it look functional.
- **Don't add models that fabricate identity.** Anything that could hand an investigator a photorealistic but fabricated face (face-swap, unconditioned face generation) is out of scope for this pipeline — see the README's [Honest Limitations](README.md#%EF%B8%8F-honest-limitations--responsible-use) section for why.
- **No real biometric data in commits, issues, or PRs.** Use synthetic, public-domain, or clearly-licensed sample images only. Never commit real people's photos, embeddings, or `data/*.db` files.
- **Keep the "human review" framing intact.** Changes that would present results as automated identifications instead of investigative leads for human review are a hard no.

## Reporting security or privacy issues

If you find a security vulnerability or a way this tool could leak/expose biometric data, please **do not open a public issue**. Instead, reach out privately via the contact info on [@Amaan9136](https://github.com/Amaan9136)'s GitHub profile.

## Questions?

Open a [discussion or issue](https://github.com/Amaan9136/periocular-criminal-identification/issues) — happy to help you get oriented.
