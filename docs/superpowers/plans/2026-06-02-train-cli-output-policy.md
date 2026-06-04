# Train CLI Output Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the train CLI to underscore-style flags, require project-scoped experiment naming, and make default output-directory collision handling deterministic.

**Architecture:** Keep config file field names unchanged, translate the new CLI surface in `tools/train.py` and `config/loader.py`, and centralize output-directory resolution in a small helper so trainer startup owns the final path decision before any artifact writes happen.

**Tech Stack:** Python 3.12, argparse, dataclasses, unittest

---

### Task 1: Lock the new CLI contract with tests

**Files:**
- Modify: `../tests/test_cls_engine_tools.py`
- Modify: `../tests/test_cls_engine_config.py`

- [ ] Add parser expectations for `--data`, `--model`, `--batch`, `--imgsz`, `--workers`, `--project`, `--name`, `--output`, and `--exist_ok`.
- [ ] Add config-loader expectations that the renamed args still override the same config fields.
- [ ] Add coverage for default `name` generation shape and `imgsz` normalization.

### Task 2: Implement parser and config-loader changes

**Files:**
- Modify: `tools/train.py`
- Modify: `src/cls_engine/config/loader.py`

- [ ] Rename CLI flags to underscore-style spellings and remove the old hyphenated aliases.
- [ ] Make `--data` and `--project` required, keep `--name` optional with `exp_YYYYMMDDHHmmss` defaulting.
- [ ] Preserve all existing config-file behavior while mapping the new CLI names onto the same dataclass fields.

### Task 3: Resolve final output directory safely

**Files:**
- Modify: `src/cls_engine/utils/paths.py`
- Modify: `src/cls_engine/engine/trainer.py`
- Modify: `../tests/test_cls_engine_config.py`

- [ ] Add a helper that computes the final output path from `project`, `name`, `output`, and `exist_ok`.
- [ ] When `--output` is explicit, fail before training if it exists and is non-empty and `exist_ok` is false.
- [ ] When `--output` is omitted, build `runs/classify/{project}/{name}` and append `_YYYYMMDDHHmmss` only if the directory already exists and is non-empty and `exist_ok` is false.
- [ ] Ensure `cfg.task.output_dir` stores the resolved final path before artifact writing and port derivation.

### Task 4: Update docs and verify

**Files:**
- Modify: `README.md`

- [ ] Update command examples to the renamed flags and project/name workflow.
- [ ] Run focused unittest coverage for parser/config/output-policy behavior.
- [ ] Run `tools/train.py --help` and confirm the visible CLI matches the spec.
