# Code provenance

This release candidate was distilled from an audited internal source snapshot. The snapshot records source commit:

```text
20bd331bdbc9026a5668e11362178e10ab7400c8
```

That snapshot is the version associated with the complete ALFWorld `afaa13` and WebShop `wfaa2` step-record exports used by the paper analyses. Environment prompts and configuration evidence were cross-checked against the corresponding internal paper-material archive.

Release cleanup changed module names, removed unrelated trainer branches and machine-specific paths, and added a framework-neutral adapter. The state/action rules and BLPO estimator were retained from the audited implementation.
