# Code provenance

This release candidate was distilled locally from the audited source snapshot at:

```text
D:/my_codex/ssh_test/BLPO_raw_20260912/code_snapshot
```

The snapshot records source commit:

```text
20bd331bdbc9026a5668e11362178e10ab7400c8
```

That snapshot is the version associated with the complete ALFWorld `afaa13` and WebShop `wfaa2` step-record exports used by the paper analyses. Environment prompt and configuration evidence was cross-checked against:

```text
D:/my_codex/ssh_test/BLPO_paper_materials_20260916/prompt_appendix
```

Release cleanup changed module names, removed unrelated trainer branches and machine-specific paths, and added a framework-neutral adapter. The state/action rules and BLPO estimator were retained from the audited implementation.
