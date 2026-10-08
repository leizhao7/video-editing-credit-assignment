# Release status

This is a source-release preparation draft. No dataset or original task instances are included. Full remote training and end-to-end benchmark reproduction have not been run from this tree.

## Recovered and organized

- Fixed PRM: all 114 files in the recorded final process-source manifest were recovered with matching SHA-256 hashes in the internal snapshot.
- Decayed PRM: all 116 files in the final process-source manifest were recovered or exactly reconstructed. Two SkyRL startup files were reconstructed using the preserved, hash-checked transformation, and their output hashes match the final run. Portable copies and patches are kept separate from the original snapshots.
- OPSD: the final turn-suggestion recipe matches its recorded source identity. Its original 20-file snapshot is retained internally; the public draft excludes the private deployment module and includes portable configuration and selected CPU tests. This is not the original reference-answer recipe.
- Outcome evaluation: the recovered fixed-PRM grader modules match the recorded source hashes; public portability and failure-handling adaptations have separate hashes. The decay changes are kept in a separate overlay. The original fixed service wrapper and stage bridge have also been identified internally, but a portable service integration remains to be completed.

## Work still needed before claiming full reproducibility

1. Restore the original reference-answer OPSD recipe with aggregate SHA-256 `43262b0844c3b778d3d9b76607c1ec0f1e298ca02781a16bcfa05a4258f45018`. The final turn-suggestion recipe must not be silently substituted.
2. Recover the complete frozen environment build/install records, including local SkyRL/vLLM patches and the OPSD installation preimages. Commit pins alone do not reconstruct a modified environment.
3. Complete public training, sandbox, grader and storage entry points with operator-supplied paths and credentials. Preserve source and rollout consistency checks while replacing historical private cloud identifiers.
4. The final quality-filtering and film-disjoint sampling procedure has not been fully reconstructed; the original 797-task selection cannot currently be reproduced from this tree.
5. Recover the earlier outcome-only grader media adapter and document differences between evaluation versions. Verify the complete grading pipeline and its aggregation against the recorded outputs before using it to regenerate tables.
6. Complete offline integration and then an explicitly configured small GPU smoke run on independently supplied or synthetic media. Passing CPU unit tests does not establish that the distributed training stack works.
7. Add the formal paper link when available. Project-authored code is licensed under [MIT](../LICENSE); third-party components retain their original licenses. The confirmed author order and contact information are recorded in `README.md` and `CITATION.cff`.

## Publication boundary

The public tree should contain code, configuration templates, generic prompts, dependency identities, source provenance, and documentation. One reviewed paper figure, `docs/assets/paper-teaser.png`, is included for presentation; it contains illustrative film thumbnails and aggregate results. Its path and exact hash are allowlisted by the release checker. The paper figure is excluded from the project's code license. The retained task schematic contains no film frames or measured results. Underlying media, raw result files, dataset manifests, movie lists, real prompts and answers, generated task bundles, raw rollouts, judge caches, checkpoints, credential files and private cloud deployment records stay outside the release. The original source snapshots and local absolute-path recovery inventory are internal working material, not a second directory to upload alongside this repository.
