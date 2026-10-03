# Anonymity Checklist

- [x] No author names.
- [x] No institutions.
- [x] No local absolute paths.
- [x] No usernames.
- [x] No private URLs.
- [x] No raw datasets.
- [x] No model weights.
- [x] No non-anonymized metadata.

Before upload, run repository-specific checks for absolute paths, usernames, private URLs, and known identifying strings. For example:

```bash
grep -R "<absolute-path-patterns>" -n supplementary/cohe_artifact || true
grep -R "<user-or-institution-patterns>" -n supplementary/cohe_artifact || true
```
