# Documentation audit and design — 2026-11-07

The [machine-readable inventory](documentation-inventory.json) enumerates every tracked/nonignored owned Markdown/RST/text document, its title, role and local link findings. Current guides listed in doc-check.json are canonical and have source-grounded revisions. Other records retain their dates/results/selection contracts and are explicitly classified as retained history, reference or release evidence. Preservation is not revalidation of past economics or runtime behavior.

## Findings and revisions

Replaced workstation-specific launch-first onboarding with safe editable development/tests. Corrected test counts, split dates, package/state paths and CPU dependency guidance. Documented direct round-driver dispatch, fixed-project assumptions, credential-schema differences, omitted data and missing general recovery guarantees. Retained locked scoring gates in program.md and added targeted supersession notes to that contract and the deployment/package research guides. Recorded the existing package/module version-string discrepancy rather than inventing consistency.

README owns onboarding/navigation; the design/source guide owns architecture, contracts, configuration and recovery limits; CI owns exact supported commands; this report owns scope/remaining limits. Source versions and release evidence remain unchanged. No application behavior or locked scorer was modified by documentation work.

This established publication checkout is the known source of this independently maintained repository. No additional native working tree is asserted to have been audited. qc_work's imported related material remains a separate ownership/data scope and is inventoried there; it is not silently overwritten by this repository's guide.

## Verification scope and self-review

`python scripts/check_docs.py` checks canonical public local links/anchors and inventories unavailable historical links separately. It uses no network, ignores fenced examples and does not execute code embedded in prose. External URLs, reference-style links, arbitrary generated HTML, omitted private data and historical performance claims are outside automated verification. The inventory is a structural/claim-disposition review, not a claim of line-by-line economic recertification for every archived analysis.

Current commands/configuration were checked against the actual entrypoints, loaders, manifests, workflow and tests. Safe CLI help is used only where the entrypoint is known to exit before account effects; quant-autoresearch's round driver is specifically excluded. Existing runtime/market/model artifacts are not created or deleted to validate prose. The complete existing offline unit suites and relevant static checks remain the acceptance baseline; no cloud account or provider acquisition test is run.

Prior REVIEW_20261107.md and release evidence remain immutable. Dated research/design plans can contain historical workstation paths, absent result/report links and unimplemented proposals. Their unavailable local references are listed in the inventory rather than fabricated or presented as public artifacts. Canonical setup/current-use instructions supersede those paths; no deployment, live-readiness or alpha claim is introduced.

The sole browser publisher serializes this phase's reviewed commits. Documentation publication needs its own exact-tree/final-SHA hosted CI acceptance; prior implementation CI success is dated evidence, not an automatic pass for this revision.

## Validation receipt

Inventory: 45 owned prose documents (4 canonical guides, 41 retained research/contract/reference/release records). All relative paths/anchors in canonical guides pass; no unavailable retained relative Markdown links were found. This does not certify external URLs, plain-text filesystem paths or omitted report/data availability. Published main `93d3237483cbe6cf86bc1b865998d092493d267e` had the same complete tree as the local pre-documentation source.

`PYTHONPATH=src /home/vic/qc_work/.venv/bin/python -m pytest -q`: 19 passed in 2.72s. Compileall for src/scripts passed. Documentation checker passed; workflow YAML was parsed/reviewed. No round-driver help or API client was executed. The stdlib checker is identical to the centrally regression-tested implementation (4 dedicated tests), and CI now runs it before package tests. Existing scoring/model/training application code and dependency metadata remain byte-identical.
