#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import statistics
try:
    from tqdm import tqdm  # type: ignore
    HAVE_TQDM = True
except Exception:
    HAVE_TQDM = False

# Try to import yaml for parsing; required for both GHA and Travis checks.
try:
    import yaml  # type: ignore
    HAVE_YAML = True
except Exception:
    HAVE_YAML = False

# Map friendly setting labels to the exact folder names under the Results directory
# Only these four folders will be scanned; others are ignored.
# Supports both hyphenated and non-hyphenated variants (normalized to hyphenated)
SETTING_DIR_MAP = {
    "Fine-Tuned": "CIgrate_Results_FineTuned",
    "Zero-Shot": "CIgrate_Results_ZeroShot",
    "ZeroShot": "CIgrate_Results_ZeroShot",
    "Few-Shot": "CIgrate_Results_FewShot",
    "FewShot": "CIgrate_Results_FewShot",
    "Fine-Tuned": "CIgrate_Results_FineTuned",
    "FineTuned": "CIgrate_Results_FineTuned",
    "CIMig": "CIMig_Results",
}

# Canonical (hyphenated) names for output
SETTING_CANONICAL = {
    "CIgrate_Results_ZeroShot": "Zero-Shot",
    "CIgrate_Results_FewShot": "Few-Shot",
    "CIgrate_Results_FineTuned": "Fine-Tuned",
    "CIMig_Results": "CIMig",
}

DIRECTIONS = {
    "03_CIgrate_Travis_to_GHA": {"target": "gha", "filename": "actions.yml"},
    "04_CIgrate_GHA_to_Travis": {"target": "travis", "filename": "travis.yml"},
    "03_CIMig_Travis_to_GHA": {"target": "gha", "filename": "actions.yml"},
    "04_CIMig_GHA_to_Travis": {"target": "travis", "filename": "travis.yml"},
}
ORIGINALS = {
    "01_Original_Travis": {"service": "travis", "filename": "travis.yml"},
    "02_Original_GHA": {"service": "gha", "filename": "actions.yml"},
}

def normalize_setting_name(folder_name: str) -> str:
    """
    Convert folder name to canonical setting label.
    Examples: CIgrate_Results_ZeroShot -> Zero-Shot, CIMig_Results -> CIMig
    """
    return SETTING_CANONICAL.get(folder_name, folder_name)

# Weights for composite score calculation (total = 1.0)
# PyYAML parsing check applies to both GHA and Travis files
# External linting uses service-specific tools: actionlint for GHA, travis lint for Travis
W_YAML_PARSE = 0.5  # PyYAML parsing (both GHA & Travis)
W_EXTERNAL = 0.5    # External linting: actionlint (GHA) or travis lint (Travis)

@dataclass
class LintResult:
    setting: str
    project: str
    direction: str  # "original", "Travis_to_GHA", "GHA_to_Travis"
    model: str      # "-", or model folder name (e.g., llama3_1_8b)
    service: str    # "gha" or "travis"
    file_path: str
    yaml_parsed: Optional[bool]
    external_ok: Optional[bool]
    yaml_score: Optional[float]
    external_score: Optional[float]
    external_tool: Optional[str]
    external_pass_count: Optional[int]
    external_total_count: Optional[int]
    # Per-tool booleans: actionlint for GHA, travis_lint for Travis
    actionlint_ok: Optional[bool]
    travis_lint_ok: Optional[bool]
    score: Optional[float]
    messages: List[str]

def which(cmd: str) -> Optional[str]:
    from shutil import which as sh_which
    return sh_which(cmd)

def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as e:
        return 127, "", str(e)

def parse_yaml(path: Path) -> Tuple[Optional[dict], Optional[str]]:
    if not HAVE_YAML:
        return None, "PyYAML not installed; YAML parse skipped"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data, None
    except Exception as e:
        return None, f"YAML parse error: {e}"

def external_lint(service: str, path: Path, repo_root: Path) -> Tuple[Optional[float], Optional[bool], Optional[str], Optional[int], Optional[int], Optional[str], Optional[bool], Optional[bool]]:
    """
    Run service-specific external linter and compute external score.
    
    Scoring system:
    - For GHA files: runs actionlint (lenient mode - only fails on syntax errors, not warnings)
    - For Travis files: runs travis lint (with fallbacks to travis-lint gem and offline checks)
    
    Actionlint lenient mode:
    - Passes: warnings about outdated action versions, deprecated features
    - Fails: syntax errors, structural issues, duplicate keys, invalid sections
    
    Returns tuple: (
        external_score (0.0 or 1.0 or None if tool not available),
        external_all_ok (True if tool passed, False if failed, None if not run),
        messages (combined output),
        pass_count (0 or 1 or None),
        total_count (1 or None),
        tools_used (tool name or None),
        actionlint_ok (bool or None),
        travis_lint_ok (bool or None)
    )
    """
    messages: List[str] = []
    actionlint_ok = None
    travis_lint_ok = None

    if service == "gha":
        # GitHub Actions: use actionlint only
        # Only fail on syntax errors, not on warnings (e.g., outdated action versions)
        actionlint_bin = which("actionlint")
        if actionlint_bin:
            rc, out, err = run_cmd([actionlint_bin, str(path)], cwd=repo_root)
            output = (out or err).lower()
            
            # Check if there are actual syntax/structural errors (not just warnings)
            # actionlint returns rc=1 for both errors and warnings
            # We only fail if there are syntax-check errors or critical issues
            has_syntax_error = "[syntax-check]" in output
            has_critical_error = rc == 1 and (
                "could not parse" in output or
                "yaml:" in output or
                "syntax error" in output or
                has_syntax_error
            )
            
            # Pass if no critical errors (warnings like outdated versions are OK)
            actionlint_ok = not has_critical_error
            
            messages.append(f"[actionlint rc={rc}] {out or err}")
            external_score = 1.0 if actionlint_ok else 0.0
            pass_count = 1 if actionlint_ok else 0
            return (
                external_score,
                actionlint_ok,
                "\n".join(messages),
                pass_count,
                1,  # total_count
                "actionlint",
                actionlint_ok,
                None  # travis_lint_ok
            )
        else:
            messages.append("[actionlint] not found; skipping")
            return (None, None, "\n".join(messages), None, None, None, None, None)

    elif service == "travis":
        # Travis CI: try travis CLI, then travis-lint gem, then offline checks
        
        # 1) Try travis CLI first
        travis_bin = which("travis")
        if travis_bin:
            rc, out, err = run_cmd([travis_bin, "lint", str(path), "--no-interactive"], cwd=repo_root)
            text = ((out or "") + "\n" + (err or "")).lower()
            # Travis CLI is deprecated and may return HTML/405; treat as inconclusive
            if ("<!doctype html>" in text) or ("method not allowed" in text) or ("outdated cli version" in text):
                messages.append(f"[travis lint inconclusive rc={rc}] {out or err}")
            else:
                messages.append(f"[travis lint rc={rc}] {out or err}")
                ok = (rc == 0) or ("is valid" in text)
                travis_lint_ok = ok
                external_score = 1.0 if ok else 0.0
                pass_count = 1 if ok else 0
                return (
                    external_score,
                    ok,
                    "\n".join(messages),
                    pass_count,
                    1,  # total_count
                    "travis lint",
                    None,  # actionlint_ok
                    travis_lint_ok
                )
        else:
            messages.append("[travis] CLI not found; trying alternatives")

        # 2) Fallback to travis-lint gem
        travis_lint_bin = which("travis-lint") or which("travis_lint")
        if travis_lint_bin:
            rc, out, err = run_cmd([travis_lint_bin, str(path)], cwd=repo_root)
            text_full = ((out or "") + "\n" + (err or "")).strip()
            text = text_full.lower()
            # Detect network/API failures and mark inconclusive
            if ("<!doctype html>" in text) or ("method not allowed" in text) or ("net::http" in text) or ("405" in text):
                messages.append(f"[travis-lint inconclusive rc={rc}] {out or err}")
            else:
                ok = (rc == 0) or ("hooray" in text) or ("is valid" in text)
                messages.append(f"[travis-lint rc={rc}] {out or err}")
                travis_lint_ok = ok
                external_score = 1.0 if ok else 0.0
                pass_count = 1 if ok else 0
                return (
                    external_score,
                    ok,
                    "\n".join(messages),
                    pass_count,
                    1,  # total_count
                    "travis-lint",
                    None,  # actionlint_ok
                    travis_lint_ok
                )
        else:
            messages.append("[travis-lint] gem not found; using offline check")

        # 3) Offline static checks as last resort
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            messages.append(f"[travis-offline] yaml load failed: {e}")
            data = None

        def is_nonempty_str_or_list(val) -> bool:
            if isinstance(val, str):
                return val.strip() != ""
            if isinstance(val, list):
                return len(val) > 0
            return False

        ok = False
        if isinstance(data, dict):
            has_language = isinstance(data.get("language"), (str, list)) and is_nonempty_str_or_list(data.get("language"))
            has_script = is_nonempty_str_or_list(data.get("script"))
            jobs = data.get("jobs") or {}
            stages = data.get("stages")
            has_jobs = isinstance(jobs, dict) and (is_nonempty_str_or_list(jobs.get("include")) or is_nonempty_str_or_list(jobs.get("script")))
            has_stages = is_nonempty_str_or_list(stages)
            ok = bool(has_language and (has_script or has_jobs or has_stages))
            messages.append("[travis-offline] " + ("passed basic sanity checks" if ok else "failed basic sanity checks (missing language and runnable section)"))
        else:
            messages.append("[travis-offline] invalid document (expected mapping)")
        
        travis_lint_ok = ok
        external_score = 1.0 if ok else 0.0
        pass_count = 1 if ok else 0
        return (
            external_score,
            ok,
            "\n".join(messages),
            pass_count,
            1,  # total_count
            "travis-offline",
            None,  # actionlint_ok
            travis_lint_ok
        )
    
    else:
        return (None, None, f"Unknown service: {service}", None, None, None, None, None)

def score_bool(b: Optional[bool]) -> Optional[float]:
    if b is None:
        return None
    return 1.0 if b else 0.0

def compute_score(yaml_parsed: Optional[bool], external_value: Optional[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    yaml_score = score_bool(yaml_parsed)
    external_score = external_value  # Already 0..1 or None
    if yaml_score is None and external_score is None:
        return None, yaml_score, external_score
    score = 0.0
    denom = 0.0
    if yaml_score is not None:
        score += yaml_score * W_YAML_PARSE
        denom += W_YAML_PARSE
    if external_score is not None:
        score += external_score * W_EXTERNAL
        denom += W_EXTERNAL
    return (score / denom if denom > 0 else None), yaml_score, external_score

def detect_model_dirs(parent: Path) -> List[Path]:
    # Any immediate subdirectory is a model folder
    return [p for p in parent.iterdir() if p.is_dir()]

def select_main_gha_file(files: List[Path]) -> Optional[Path]:
    """
    Mirrors provided Java logic:
      1) Scan in order; set main file to the LAST path whose string contains any of
         the keywords: build, compile, main, maven, ci (case-sensitive substring).
      2) If none matched, pick the first file with the strictly largest size (>).
         If list is empty, return None.
    """
    main_gh_file: Optional[Path] = None
    for f in files:
        s = str(f)
        if ("build" in s) or ("compile" in s) or ("main" in s) or ("maven" in s) or ("ci" in s):
            main_gh_file = f
    if main_gh_file is None:
        max_size = 0
        for f in files:
            try:
                size = f.stat().st_size
            except Exception:
                continue
            if size > max_size:
                main_gh_file = f
                max_size = int(size)
    return main_gh_file

def lint_file(setting: str, project: str, direction: str, model: str, service: str, path: Path) -> List[LintResult]:
    msgs: List[str] = []
    repo_root = path.parent

    yaml_parsed = None
    external_ok = None
    tool_used = None

    parse_data, parse_err = parse_yaml(path)
    if parse_err:
        msgs.append(parse_err)
        yaml_parsed = False if HAVE_YAML else None
    else:
        yaml_parsed = True

    ext_score, ext_all_ok, ext_msg, ext_pass, ext_total, tools_used, actionlint_ok, travis_ok = external_lint(service, path, repo_root)
    external_ok = ext_all_ok
    tool_used = tools_used
    if ext_msg:
        msgs.append(ext_msg)

    composite, yaml_score, external_score = compute_score(yaml_parsed, ext_score)

    return [LintResult(
        setting=setting,
        project=project,
        direction=direction,
        model=model,
        service=service,
        file_path=str(path),
        yaml_parsed=yaml_parsed,
        external_ok=external_ok,
        yaml_score=yaml_score,
        external_score=external_score,
        external_tool=tool_used,
        external_pass_count=ext_pass,
        external_total_count=ext_total,
        actionlint_ok=actionlint_ok,
        travis_lint_ok=travis_ok,
        score=round(composite, 4) if composite is not None else None,
        messages=msgs
    )]

def detect_projects(base: Path) -> Dict[str, List[Tuple[str, Path, Optional[str]]]]:
    """
    Scan the base Results directory and build a mapping of project -> list of (setting_label, setting_dir, model_folder).
    
    For normal settings (Zero-Shot, Few-Shot, CIMig):
        structure: base/CIgrate_Results_ZeroShot/Project1/...
        returns: ("Zero-Shot", setting_dir, None)
    
    For Fine-Tuned:
        structure: base/CIgrate_Results_FineTuned/codegemma_7b_it_finetune_travis_to_gha_lora/Project1/...
        returns: ("Fine-Tuned", setting_dir, "codegemma_7b_it_finetune_travis_to_gha_lora")
    
    Only the configured subfolders in SETTING_DIR_MAP are considered.
    """
    projects: Dict[str, List[Tuple[str, Path, Optional[str]]]] = {}
    
    # Get unique folder names (since we now have duplicate keys for variants)
    unique_folders = set(SETTING_DIR_MAP.values())
    
    for folder_name in unique_folders:
        sdir = base / folder_name
        if not sdir.exists() or not sdir.is_dir():
            continue
        
        setting_label = normalize_setting_name(folder_name)
        
        # Special handling for Fine-Tuned (nested model folders)
        if folder_name == "CIgrate_Results_FineTuned":
            for set_folder in sdir.iterdir():
                if not set_folder.is_dir():
                    continue
                # model_folder is like "codegemma_7b_it_finetune_travis_to_gha_lora"
                for proj_dir in set_folder.iterdir():
                    if proj_dir.is_dir():
                        projects.setdefault(proj_dir.name, []).append((setting_label, sdir, set_folder.name))
        else:
            # Normal structure: direct project folders
            for proj_dir in sdir.iterdir():
                if proj_dir.is_dir():
                    projects.setdefault(proj_dir.name, []).append((setting_label, sdir, None))
    
    return projects

def gather_targets(setting_label: str, setting_dir: Path, project: str, include_originals: bool) -> List[LintResult]:
    results: List[LintResult] = []

    if include_originals:
        for orig_dirname, meta in ORIGINALS.items():
            d = setting_dir / project / orig_dirname
            if not d.exists():
                continue
            # For GHA originals, select a main file per rules, else use expected filename
            if meta["service"] == "gha":
                gha_candidates = list(d.rglob("*.yml")) + list(d.rglob("*.yaml"))
                f = select_main_gha_file(gha_candidates) if gha_candidates else None
                if f is None:
                    f = d / meta["filename"]
            else:
                # For Travis originals, use expected file if present, else first *.yml or *.yaml
                expected = d / meta["filename"]
                if expected.exists():
                    f = expected
                else:
                    travis_candidates = list(d.rglob("*.yml")) + list(d.rglob("*.yaml"))
                    f = travis_candidates[0] if travis_candidates else None
            if not f or not f.exists():
                continue
            results.extend(lint_file(
                setting=setting_label,
                project=project,
                direction="original",
                model="original",
                service=meta["service"],
                path=f
            ))

    for direction_dirname, meta in DIRECTIONS.items():
        d = setting_dir / project / direction_dirname
        if not d.exists():
            continue
        for model_dir in detect_model_dirs(d):
            files: List[Path] = []
            if meta["target"] == "gha":
                # For GHA generation, choose a single main actions file per model
                candidates = list(model_dir.rglob("*.yml")) + list(model_dir.rglob("*.yaml"))
                main_file = select_main_gha_file(candidates) if candidates else None
                if main_file and main_file.exists():
                    files = [main_file]
            else:
                # For Travis generation, use expected filename if present, else the first .yml/.yaml found
                expected = model_dir / meta["filename"]
                if expected.exists():
                    files = [expected]
                else:
                    candidates = list(model_dir.rglob("*.yml")) + list(model_dir.rglob("*.yaml"))
                    if candidates:
                        files = [candidates[0]]
            if not files:
                continue
            direction_label = "Travis_to_GHA" if "Travis_to_GHA" in direction_dirname else "GHA_to_Travis"
            for f in files:
                results.extend(lint_file(
                    setting=setting_label,
                    project=project,
                    direction=direction_label,
                    model=model_dir.name,
                    service=meta["target"],
                    path=f
                ))
    return results

def discover_all_targets(base: Path, include_originals: bool) -> List[Tuple[str, str, str, str, str, Path]]:
    """
    Discover all lint targets and return a list of tuples:
    (setting_label, project, direction, model, service, path)
    where direction is one of: original, Travis_to_GHA, GHA_to_Travis (normalized to lowercase in output)
    and model is 'original' for originals.
    Uses the same selection rules as gather_targets but only collects metadata.
    """
    targets: List[Tuple[str, str, str, str, str, Path]] = []
    projects = detect_projects(base)
    
    for project, sdir_entries in sorted(projects.items(), key=lambda x: x[0]):
        for setting_label, setting_dir, model_folder_name in sdir_entries:
            # Determine base project path based on Fine-Tuned vs normal structure
            if model_folder_name:
                # Fine-Tuned: setting_dir/model_folder/project/
                project_base = setting_dir / model_folder_name / project
            else:
                # Normal: setting_dir/project/
                project_base = setting_dir / project
            
            # directions (and attach the appropriate original per direction)
            for direction_dirname, meta in DIRECTIONS.items():
                d = project_base / direction_dirname
                if not d.exists():
                    continue
                
                # Normalize direction label to lowercase
                direction_label = "travis_to_gha" if "Travis_to_GHA" in direction_dirname else "gha_to_travis"
                
                # Attach original for this direction if requested
                if include_originals:
                    if "Travis_to_GHA" in direction_dirname:
                        # Use original GHA
                        orig_dir = project_base / "02_Original_GHA"
                        f: Optional[Path] = None
                        if orig_dir.exists():
                            gha_candidates = list(orig_dir.rglob("*.yml")) + list(orig_dir.rglob("*.yaml"))
                            f = select_main_gha_file(gha_candidates) if gha_candidates else None
                            if f is None:
                                f = orig_dir / ORIGINALS["02_Original_GHA"]["filename"]
                        if f and f.exists():
                            model_label = "CIMig" if setting_label == "CIMig" else "original"
                            targets.append((setting_label, project, direction_label, model_label, "gha", f))
                    else:
                        # GHA_to_Travis uses original Travis
                        orig_dir = project_base / "01_Original_Travis"
                        f = None
                        if orig_dir.exists():
                            expected = orig_dir / ORIGINALS["01_Original_Travis"]["filename"]
                            if expected.exists():
                                f = expected
                            else:
                                travis_candidates = list(orig_dir.rglob("*.yml")) + list(orig_dir.rglob("*.yaml"))
                                f = travis_candidates[0] if travis_candidates else None
                        if f and f.exists():
                            model_label = "CIMig" if setting_label == "CIMig" else "original"
                            targets.append((setting_label, project, direction_label, model_label, "travis", f))
                
                # CIMig has files directly in direction folder, others have model subdirectories
                if setting_label == "CIMig":
                    # CIMig: files are directly in the direction folder
                    if meta["target"] == "gha":
                        candidates = list(d.glob("*.yml")) + list(d.glob("*.yaml"))
                        main_file = select_main_gha_file(candidates) if candidates else None
                        files = [main_file] if main_file and main_file.exists() else []
                    else:
                        candidates = list(d.glob("*.yml")) + list(d.glob("*.yaml"))
                        files = [candidates[0]] if candidates else []
                    
                    for f in files:
                        targets.append((setting_label, project, direction_label, "CIMig", meta["target"], f))
                else:
                    # Other settings: iterate through model subdirectories
                    for model_dir in detect_model_dirs(d):
                        if meta["target"] == "gha":
                            candidates = list(model_dir.rglob("*.yml")) + list(model_dir.rglob("*.yaml"))
                            main_file = select_main_gha_file(candidates) if candidates else None
                            files = [main_file] if main_file and main_file.exists() else []
                        else:
                            expected = model_dir / meta["filename"]
                            if expected.exists():
                                files = [expected]
                            else:
                                candidates = list(model_dir.rglob("*.yml")) + list(model_dir.rglob("*.yaml"))
                                files = [candidates[0]] if candidates else []
                        if not files:
                            continue
                        
                        for f in files:
                            # Extract model name from subfolder (e.g., codegemma_7b_it_finetune_lora)
                            model_label = model_dir.name
                            targets.append((setting_label, project, direction_label, model_label, meta["target"], f))
    
    return targets

# ---------- Writers & Aggregations ----------

def write_csv(results: List[LintResult], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "setting","project","direction","model","service","file_path",
            "yaml_parsed","external_ok",
            "yaml_score","external_score",
            "external_tool","external_pass_count","external_total_count","actionlint_ok","travis_lint_ok",
            "score","messages"
        ])
        for r in results:
            w.writerow([
                r.setting, r.project, r.direction, r.model, r.service, r.file_path,
                r.yaml_parsed, r.external_ok,
                r.yaml_score, r.external_score,
                r.external_tool, r.external_pass_count, r.external_total_count, r.actionlint_ok, r.travis_lint_ok,
                r.score,
                " | ".join([m.replace("\n", " / ") for m in r.messages])[:5000]
            ])

def mean_or_none(xs: List[Optional[float]]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)

def pass_rate(xs: List[Optional[bool]]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    if not vals:
        return None
    return sum(1 for x in vals if x) / len(vals)

def agg(rows: List[LintResult], keys: List[str]) -> List[Dict[str, object]]:
    from collections import defaultdict
    bucket = defaultdict(list)
    for r in rows:
        key = tuple(getattr(r, k) for k in keys)
        bucket[key].append(r)
    out = []
    for key, items in bucket.items():
        rec = {k:v for k, v in zip(keys, key)}
        rec["n_files"] = len(items)
        def safe_round(v: Optional[float]) -> Optional[float]:
            return round(v, 4) if v is not None else None

        rec["score_mean"] = safe_round(mean_or_none([i.score for i in items])) if items else None
        rec["yaml_pass_rate"] = safe_round(pass_rate([i.yaml_parsed for i in items])) if items else None
        rec["external_pass_rate"] = safe_round(pass_rate([i.external_ok for i in items])) if items else None
        rec["yaml_score_mean"] = safe_round(mean_or_none([i.yaml_score for i in items])) if items else None
        rec["external_score_mean"] = safe_round(mean_or_none([i.external_score for i in items])) if items else None
        out.append(rec)
    return out

def agg_external_coverage(rows: List[LintResult]) -> List[Dict[str, object]]:
    from collections import defaultdict
    bucket = defaultdict(list)
    for r in rows:
        key = (r.setting, r.direction, r.service)
        bucket[key].append(r)
    out: List[Dict[str, object]] = []
    for (setting, direction, service), items in bucket.items():
        total = len(items)
        any_ext = sum(1 for i in items if i.external_total_count and i.external_total_count > 0)
        all_ok = sum(1 for i in items if i.external_ok is True)
        tools = set()
        for i in items:
            if i.external_tool:
                for t in i.external_tool.split(","):
                    tools.add(t.strip())
        out.append({
            "setting": setting,
            "direction": direction,
            "service": service,
            "n_files": total,
            "n_with_external": any_ext,
            "external_all_ok_files": all_ok,
            "tools_seen": ", ".join(sorted(tools)),
        })
    return out

def write_agg_csv(rows: List[Dict[str, object]], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            f.write("empty\n")
        return
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def quartiles(values: List[float]) -> Tuple[float, float, float, float, float]:
    """Return (min, q1, q2, median, max). q2 is the second quartile from statistics.quantiles with inclusive method."""
    if not values:
        raise ValueError("quartiles() requires non-empty values")
    vals = sorted(values)
    min_v = vals[0]
    max_v = vals[-1]
    # Use inclusive method to better match intuitive quartiles for small samples
    try:
        q1, q2, q3 = statistics.quantiles(vals, n=4, method="inclusive")
    except TypeError:
        # Fallback for older Python versions without 'method' kwarg
        q1, q2, q3 = statistics.quantiles(vals, n=4)
    med = statistics.median(vals)
    return min_v, q1, q2, med, max_v

def agg_stats(rows: List[LintResult], keys: List[str], value_attr: str = "score") -> List[Dict[str, object]]:
    """
    Compute statistics for a given value attribute, grouped by specified keys.
    Also includes component score statistics when value_attr='score'.
    """
    from collections import defaultdict
    bucket = defaultdict(list)
    for r in rows:
        key = tuple(getattr(r, k) for k in keys)
        val = getattr(r, value_attr)
        if val is not None:
            bucket[key].append(r)  # Store full record instead of just value
    
    out: List[Dict[str, object]] = []
    for key, items in bucket.items():
        rec = {k: v for k, v in zip(keys, key)}
        
        # Main value statistics
        vals = [float(getattr(r, value_attr)) for r in items if getattr(r, value_attr) is not None]
        rec["n"] = len(vals)
        
        if vals:
            mn, q1, q2, med, mx = quartiles(vals)
            rec.update({
                "min": round(mn, 4),
                "q1": round(q1, 4),
                "q2": round(q2, 4),
                "median": round(med, 4),
                "max": round(mx, 4),
                "mean": round(sum(vals)/len(vals), 4),
            })
        
        # Add component averages when analyzing composite scores
        if value_attr == "score":
            def avg_score(attr: str) -> Optional[float]:
                scores = [getattr(r, attr) for r in items if getattr(r, attr) is not None]
                return round(sum(scores) / len(scores), 4) if scores else None
            
            rec["yaml_score_avg"] = avg_score("yaml_score")
            rec["external_score_avg"] = avg_score("external_score")
            
            # Calculate composite from component averages
            if all(rec.get(k) is not None for k in ["yaml_score_avg", "external_score_avg"]):
                rec["composite_from_avgs"] = round(
                    rec["yaml_score_avg"] * 0.5 + rec["external_score_avg"] * 0.5,
                    4
                )
            else:
                rec["composite_from_avgs"] = None
        
        out.append(rec)
    return out

def best_by_median(rows: List[LintResult], group_keys: List[str], choice_key: str = "model") -> List[Dict[str, object]]:
    """Within each (setting, direction) group, pick the 'choice_key' with highest median score.
    Ties are broken by higher q1, then higher min, then higher mean, then by name.
    """
    # First compute stats per (group_keys + [choice_key])
    stats_rows = agg_stats(rows, keys=group_keys + [choice_key], value_attr="score")
    from collections import defaultdict
    grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for rec in stats_rows:
        gkey = tuple(rec[k] for k in group_keys)
        grouped[gkey].append(rec)
    winners: List[Dict[str, object]] = []
    for gkey, candidates in grouped.items():
        # Filter only those with median present
        cands = [c for c in candidates if "median" in c]
        if not cands:
            # keep an empty marker
            winners.append({**{k: v for k, v in zip(group_keys, gkey)}, choice_key: None})
            continue
        def rank_key(c: Dict[str, object]):
            return (
                c.get("median", -1.0),
                c.get("q1", -1.0),
                c.get("min", -1.0),
                c.get("mean", -1.0),
                str(c.get(choice_key, ""))
            )
        best = sorted(cands, key=rank_key, reverse=True)[0]
        winners.append(best)
    return winners

def rank_models_by_setting_direction(rows: List[LintResult]) -> List[Dict[str, object]]:
    """Return all models per (setting,direction), ranked by median desc then q1, min, mean.
    Useful to see alternatives beyond just the best."""
    stats_rows = agg_stats(rows, keys=["setting", "direction", "model"], value_attr="score")
    def rkey(rec: Dict[str, object]):
        return (
            rec.get("median", -1.0),
            rec.get("q1", -1.0),
            rec.get("min", -1.0),
            rec.get("mean", -1.0),
            str(rec.get("model", ""))
        )
    # Group by (setting,direction)
    from collections import defaultdict
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for rec in stats_rows:
        grouped[(str(rec["setting"]), str(rec["direction"]))].append(rec)
    out: List[Dict[str, object]] = []
    for (setting, direction), items in grouped.items():
        for rec in sorted(items, key=rkey, reverse=True):
            out.append(rec)
    return out

def agg_by_setting_direction(rows: List[LintResult]) -> List[Dict[str, object]]:
    """
    Aggregate performance across all models within each (setting, direction, service).
    Shows setting-level comparison: Fine-Tuned vs Few-Shot vs Zero-Shot vs CIMig.
    
    Returns:
    - Component score averages (yaml, external)
    - Composite score statistics (n, min, q1, median, max, mean)
    - Best model name
    """
    from collections import defaultdict
    
    # Group by (setting, direction, service)
    bucket = defaultdict(list)
    for r in rows:
        if r.score is not None:
            key = (r.setting, r.direction, r.service)
            bucket[key].append(r)
    
    out: List[Dict[str, object]] = []
    for (setting, direction, service), items in bucket.items():
        scores = [r.score for r in items if r.score is not None]
        
        # Helper for score averages
        def avg_score(attr: str) -> Optional[float]:
            vals = [getattr(r, attr) for r in items if getattr(r, attr) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None
        
        # Component averages
        yaml_score_avg = avg_score("yaml_score")
        external_score_avg = avg_score("external_score")
        
        # Composite score statistics
        if scores:
            mn, q1, q2, med, mx = quartiles([float(s) for s in scores])
            mean_score = sum(scores) / len(scores)
            
            # Find best model by median score within this group
            model_scores = defaultdict(list)
            for r in items:
                if r.score is not None:
                    model_scores[r.model].append(r.score)
            
            best_model = None
            best_median = -1.0
            for model, model_vals in model_scores.items():
                if model_vals:
                    _, _, _, model_med, _ = quartiles([float(v) for v in model_vals])
                    if model_med > best_median:
                        best_median = model_med
                        best_model = model
            
            # Calculate composite from component averages
            if yaml_score_avg is not None and external_score_avg is not None:
                composite_calculated = round(
                    yaml_score_avg * 0.5 + external_score_avg * 0.5,
                    4
                )
            else:
                composite_calculated = None
            
            out.append({
                "setting": setting,
                "direction": direction,
                "service": service,
                "n_files": len(scores),
                # Component score averages
                "yaml_score_avg": yaml_score_avg,
                "external_score_avg": external_score_avg,
                # Formula: composite = yaml*0.5 + external*0.5
                "composite_from_avgs": composite_calculated,
                # Composite score statistics (from individual files)
                "min": round(mn, 4) if isinstance(mn, float) else mn,
                "q1": round(q1, 4) if isinstance(q1, float) else q1,
                "q2": round(q2, 4) if isinstance(q2, float) else q2,
                "median": round(med, 4) if isinstance(med, float) else med,
                "max": round(mx, 4) if isinstance(mx, float) else mx,
                "mean": round(mean_score, 4),
                "best_model": best_model,
            })
    
    return out

def agg_tools_by_project_direction(rows: List[LintResult]) -> List[Dict[str, object]]:
    """
    For each (setting, direction, project, service), compute pass rates and score averages.
    
    Shows both scoring components:
    1. PyYAML parsing (0.5 weight) - yaml_parsed, yaml_score
    2. External linting (0.5 weight) - actionlint_ok/travis_lint_ok, external_score
    """
    from collections import defaultdict
    bucket = defaultdict(list)
    for r in rows:
        key = (r.setting, r.direction, r.project, r.service)
        bucket[key].append(r)
    out: List[Dict[str, object]] = []
    for (setting, direction, project, service), items in bucket.items():
        n = len(items)
        
        # Helper for pass rates
        def pr(values: List[Optional[bool]]) -> Optional[float]:
            vals = [v for v in values if v is not None]
            if not vals:
                return None
            return round(sum(1 for v in vals if v) / len(vals), 4)
        
        # Helper for score averages
        def avg(values: List[Optional[float]]) -> Optional[float]:
            vals = [v for v in values if v is not None]
            if not vals:
                return None
            return round(sum(vals) / len(vals), 4)
        
        # Component 1: PyYAML (0.5 weight)
        yaml_pass_rate = pr([i.yaml_parsed for i in items])
        yaml_score_avg = avg([i.yaml_score for i in items])
        
        # Component 2: External linting (0.5 weight)
        actionlint_pass_rate = pr([i.actionlint_ok for i in items])
        travis_lint_pass_rate = pr([i.travis_lint_ok for i in items])
        external_score_avg = avg([i.external_score for i in items])
        
        # Sample messages (first few, truncated)
        msgs = []
        for i in items[:3]:
            if i.messages:
                msgs.append("; ".join(i.messages)[:200])
        
        out.append({
            "setting": setting,
            "direction": direction,
            "project": project,
            "service": service,
            "n_files": n,
            # Component scores and pass rates
            "yaml_pass_rate": yaml_pass_rate,
            "yaml_score_avg": yaml_score_avg,
            "external_score_avg": external_score_avg,
            "actionlint_pass_rate": actionlint_pass_rate,
            "travis_lint_pass_rate": travis_lint_pass_rate,
            "messages_sample": " | ".join(msgs)
        })
    return out

def agg_tools_by_setting_direction_model(rows: List[LintResult]) -> List[Dict[str, object]]:
    """
    For each (setting, direction, model, service), compute pass rates and score statistics.
    
    Pass rates and scores for both scoring components:
    1. PyYAML parsing (0.5 weight) - yaml_parsed, yaml_score
    2. External linting (0.5 weight) - actionlint_ok/travis_lint_ok, external_score
    
    Formula: composite_score = yaml_score*0.5 + external_score*0.5
    """
    from collections import defaultdict
    bucket = defaultdict(list)
    for r in rows:
        key = (r.setting, r.direction, r.model, r.service)
        bucket[key].append(r)
    out: List[Dict[str, object]] = []
    for (setting, direction, model, service), items in bucket.items():
        n = len(items)
        
        # Helper for pass rates (bool -> rate)
        def pr(values: List[Optional[bool]]) -> Optional[float]:
            vals = [v for v in values if v is not None]
            if not vals:
                return None
            return round(sum(1 for v in vals if v) / len(vals), 4)
        
        # Helper for score averages (float -> mean)
        def avg(values: List[Optional[float]]) -> Optional[float]:
            vals = [v for v in values if v is not None]
            if not vals:
                return None
            return round(sum(vals) / len(vals), 4)
        
        # Component 1: PyYAML (0.5 weight)
        yaml_pass_rate = pr([i.yaml_parsed for i in items])
        yaml_score_avg = avg([i.yaml_score for i in items])
        
        # Component 2: External linting (0.5 weight)
        actionlint_pass_rate = pr([i.actionlint_ok for i in items])
        travis_lint_pass_rate = pr([i.travis_lint_ok for i in items])
        external_score_avg = avg([i.external_score for i in items])
        
        # Composite score statistics
        scores = [i.score for i in items if i.score is not None]
        if scores:
            mn, q1, q2, med, mx = quartiles([float(s) for s in scores])
            mean_v = round(sum(scores)/len(scores), 4)
        else:
            mn = q1 = q2 = med = mx = mean_v = None
        
        # Calculated composite from component averages (for verification)
        if yaml_score_avg is not None and external_score_avg is not None:
            composite_calculated = round(
                yaml_score_avg * 0.5 + external_score_avg * 0.5,
                4
            )
        else:
            composite_calculated = None
        
        out.append({
            "setting": setting,
            "direction": direction,
            "model": model,
            "service": service,
            "n_files": n,
            # Component scores and pass rates
            "yaml_pass_rate": yaml_pass_rate,
            "yaml_score_avg": yaml_score_avg,
            "external_score_avg": external_score_avg,
            "actionlint_pass_rate": actionlint_pass_rate,
            "travis_lint_pass_rate": travis_lint_pass_rate,
            # Formula: composite = yaml*0.5 + external*0.5
            "composite_from_avgs": composite_calculated,
            # Composite score statistics (from individual file scores)
            "min": round(mn,4) if isinstance(mn,float) else mn,
            "q1": round(q1,4) if isinstance(q1,float) else q1,
            "q2": round(q2,4) if isinstance(q2,float) else q2,
            "median": round(med,4) if isinstance(med,float) else med,
            "max": round(mx,4) if isinstance(mx,float) else mx,
            "mean": mean_v,
        })
    return out

def main():
    ap = argparse.ArgumentParser(description="Generate linting scores for CIgrate outputs (requires PyYAML)")
    ap.add_argument(
        "--base-dir",
        type=str,
        default="/Users/nhossaincse/Documents/Personal/Study Material/MSc in AMODBDA/Thesis/Playground/Results",
        help=(
            "Base Results directory. Only these subfolders are scanned: "
            f"{', '.join(sorted(set(SETTING_DIR_MAP.values())))}"
        ),
    )
    ap.add_argument("--out-dir", type=str, default="./_lint_results", help="Directory to write CSV/JSON summaries (timestamp added to filenames by default)")
    ap.add_argument("--no-timestamp", action="store_true", help="Do not add any timestamp (filenames without suffix)")
    ap.add_argument("--timestamp-style", choices=["suffix", "subdir"], default="suffix", help="Where to place the timestamp: append to filenames (suffix) or as a subdirectory (subdir)")
    ap.add_argument("--skip-originals", action="store_true", help="Skip linting originals")
    args = ap.parse_args()

    if not HAVE_YAML:
        raise SystemExit("PyYAML is required. Please install with: python -m pip install pyyaml")

    base = Path(args.base_dir).resolve()
    outdir_base = Path(args.out_dir).resolve()
    outdir_base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    use_suffix = (not args.no_timestamp) and (args.timestamp_style == "suffix")
    use_subdir = (not args.no_timestamp) and (args.timestamp_style == "subdir")
    outdir = outdir_base / stamp if use_subdir else outdir_base
    outdir.mkdir(parents=True, exist_ok=True)

    def make_path(name: str, ext: str) -> Path:
        if use_subdir:
            return outdir / f"{name}.{ext}"
        if use_suffix:
            return outdir / f"{name}_{stamp}.{ext}"
        return outdir / f"{name}.{ext}"

    # Discover all targets first (for progress reporting)
    targets = discover_all_targets(base, include_originals=(not args.skip_originals))
    total = len(targets)
    # Print a quick summary so users know it's working
    print(f"Discovered {total} lint target(s) under {base} (include_originals={'yes' if not args.skip_originals else 'no'})")

    # Simple textual progress fallback when tqdm is unavailable
    def simple_progress(it, total_count: int):
        if total_count <= 0:
            # nothing to iterate
            for x in it:
                yield x
            return
        # Update roughly 10 times across the run
        step = max(1, total_count // 10)
        for idx, item in enumerate(it, 1):
            if idx == 1 or idx == total_count or (idx % step == 0):
                print(f"Linting {idx}/{total_count} ...", flush=True)
            yield item

    iter_fn = (lambda it: tqdm(it, total=total, desc="Linting", position=0, leave=True)) if HAVE_TQDM and total > 0 else (lambda it: simple_progress(it, total))

    all_results: List[LintResult] = []
    for setting_label, project, direction, model, service, path in iter_fn(targets):
        all_results.extend(lint_file(setting_label, project, direction, model, service, path))

    # Detailed CSV
    detailed_csv = make_path("lint_detailed", "csv")
    write_csv(all_results, detailed_csv)

    # Aggregations
    # 1) Model-wise within setting/direction/service
    agg_model = agg(all_results, keys=["setting", "direction", "model", "service"])
    write_agg_csv(agg_model, make_path("agg_by_setting_direction_model_service", "csv"))

    # 2) Setting-wise per project (overall across directions/models/services)
    agg_project_setting = agg(all_results, keys=["project", "setting"])
    write_agg_csv(agg_project_setting, make_path("agg_by_project_and_setting", "csv"))

    # 3) Model-wise per project & setting & direction
    agg_project_setting_model_dir = agg(all_results, keys=["project", "setting", "direction", "model", "service"])
    write_agg_csv(agg_project_setting_model_dir, make_path("agg_by_project_setting_direction_model_service", "csv"))

    # 4) Overall per project
    agg_project = agg(all_results, keys=["project"])
    write_agg_csv(agg_project, make_path("agg_overall_by_project", "csv"))

    # 5) Stats (min/max/quartiles) per setting+direction+model
    stats_sdm = agg_stats(all_results, keys=["setting", "direction", "model"], value_attr="score")
    if stats_sdm:
        write_agg_csv(stats_sdm, make_path("stats_by_setting_direction_model", "csv"))

    # 6) Best model per (setting, direction) by median score
    best_med = best_by_median(all_results, group_keys=["setting", "direction"], choice_key="model")
    if best_med:
        write_agg_csv(best_med, make_path("best_model_by_setting_direction", "csv"))

    # 7) External tool coverage summary
    external_cov = agg_external_coverage(all_results)
    if external_cov:
        write_agg_csv(external_cov, make_path("external_coverage_by_setting_direction_service", "csv"))

    # 8) Full rankings per (setting,direction)
    ranked_models = rank_models_by_setting_direction(all_results)
    if ranked_models:
        write_agg_csv(ranked_models, make_path("ranked_models_by_setting_direction", "csv"))

    # 9) Per-project tool pass rates and means
    tools_by_proj = agg_tools_by_project_direction(all_results)
    if tools_by_proj:
        write_agg_csv(tools_by_proj, make_path("tools_by_project_direction", "csv"))

    # 10) Model-level tool scorecard
    tools_by_model = agg_tools_by_setting_direction_model(all_results)
    if tools_by_model:
        write_agg_csv(tools_by_model, make_path("scorecard_by_setting_direction_model", "csv"))

    # 11) Setting-level summary (aggregated across all models)
    setting_summary = agg_by_setting_direction(all_results)
    if setting_summary:
        write_agg_csv(setting_summary, make_path("summary_by_setting_direction", "csv"))

    print(f"Wrote: {detailed_csv}")
    print(f"Wrote all aggregation CSVs to: {outdir}")

if __name__ == "__main__":
    main()
