# Changelog

## 0.1.1

- Make the JavaScript score-v3 upgrade match Python's spec mapping. Keep the
  editor's recovery metadata in its import representation, so v1/v2 scores
  still save unchanged until an edit requires v3.
- Preserve unknown top-level v1/v2 metadata named `chain`, `seed`, `lanes`,
  `legacy` or `legacyTop` through saves and undo, alongside pattern metadata.
- Compare 200 seeded v1/v2 score upgrades and lossless editor round trips
  against Python, including unknown metadata and colliding field names.
