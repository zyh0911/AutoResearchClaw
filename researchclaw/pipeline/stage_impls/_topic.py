"""Stages 1-2: Topic initialization and problem decomposition."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.hardware import detect_hardware, ensure_torch_available
from researchclaw.llm.client import LLMClient
from researchclaw.pipeline._domain import _detect_domain
from researchclaw.pipeline._helpers import (
    StageResult,
    _get_evolution_overlay,
    _read_prior_artifact,
    _safe_json_loads,
    _utcnow_iso,
)
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw.prompts import PromptManager

logger = logging.getLogger(__name__)


def _execute_topic_init(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    topic = config.research.topic
    domains = (
        ", ".join(config.research.domains) if config.research.domains else "general"
    )
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "topic_init")
        sp = _pm.for_stage(
            "topic_init",
            evolution_overlay=_overlay,
            topic=topic,
            domains=domains,
            project_name=config.project.name,
            quality_threshold=config.research.quality_threshold,
        )
        resp = llm.chat(
            [{"role": "user", "content": sp.user}],
            system=sp.system,
        )
        goal_md = resp.content
    else:
        goal_md = f"""# Research Goal

## Topic
{topic}

## Scope
Investigate the topic with emphasis on reproducible methods and measurable outcomes.

## SMART Goal
- Specific: Build a focused research plan for {topic}
- Measurable: Produce literature shortlist, hypotheses, experiment plan, and final paper
- Achievable: Complete through staged pipeline with gate checks
- Relevant: Aligned with project {config.project.name}
- Time-bound: Constrained by pipeline execution budget

## Constraints
- Quality threshold: {config.research.quality_threshold}
- Daily paper target: {config.research.daily_paper_count}

## Success Criteria
- At least 2 falsifiable hypotheses
- Executable experiment code and results analysis
- Revised paper passing quality gate

## Generated
{_utcnow_iso()}
"""
    (stage_dir / "goal.md").write_text(goal_md, encoding="utf-8")

    # --- Hardware detection (GPU / MPS / CPU) ---
    # When using ssh_remote, detect hardware on the remote host instead of locally
    _ssh_cfg = config.experiment.ssh_remote if config.experiment.mode == "ssh_remote" else None
    hw = detect_hardware(ssh_config=_ssh_cfg)
    (stage_dir / "hardware_profile.json").write_text(
        json.dumps(hw.to_dict(), indent=2), encoding="utf-8"
    )
    if hw.warning:
        logger.warning("Hardware advisory: %s", hw.warning)
    else:
        logger.info("Hardware detected: %s (%s, %s MB VRAM)", hw.gpu_name, hw.gpu_type, hw.vram_mb)

    # --- Optionally ensure PyTorch is available ---
    if hw.has_gpu and config.experiment.mode == "sandbox":
        torch_ok = ensure_torch_available(config.experiment.sandbox.python_path, hw.gpu_type)
        if torch_ok:
            logger.info("PyTorch is available for sandbox experiments")
        else:
            logger.warning("PyTorch could not be installed; sandbox will use CPU-only packages")
    elif hw.has_gpu and config.experiment.mode == "docker":
        logger.info("Docker sandbox: PyTorch pre-installed in container image")

    return StageResult(
        stage=Stage.TOPIC_INIT,
        status=StageStatus.DONE,
        artifacts=("goal.md", "hardware_profile.json"),
        evidence_refs=("stage-01/goal.md", "stage-01/hardware_profile.json"),
    )


def _execute_problem_decompose(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    goal_text = _read_prior_artifact(run_dir, "goal.md") or ""
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "problem_decompose")
        sp = _pm.for_stage(
            "problem_decompose",
            evolution_overlay=_overlay,
            topic=config.research.topic,
            goal_text=goal_text,
        )
        resp = llm.chat(
            [{"role": "user", "content": sp.user}],
            system=sp.system,
        )
        body = resp.content
    else:
        body = f"""# Problem Decomposition

## Source
Derived from `goal.md` for topic: {config.research.topic}

## Sub-questions
1. Which problem settings and benchmarks define current SOTA?
2. Which methodological gaps remain unresolved?
3. Which hypotheses are testable under realistic constraints?
4. Which datasets and metrics best discriminate method quality?
5. Which failure modes can invalidate expected gains?

## Priority Ranking
1. Problem framing and benchmark setup
2. Gap identification and hypothesis formulation
3. Experiment and metric design
4. Failure analysis and robustness checks

## Risks
- Ambiguous task definition
- Dataset leakage or metric mismatch

## Generated
{_utcnow_iso()}
"""
    (stage_dir / "problem_tree.md").write_text(body, encoding="utf-8")

    # IMP-35: Topic/title quality pre-evaluation + auto-refinement
    # If the topic is too broad (score < 6), generate specific sub-topics and pick the best.
    if llm is not None:
        try:
            _domain_label = _detect_domain(config.research.topic, config.research.domains)[1]
            _eval_resp = llm.chat(
                [
                    {
                        "role": "user",
                        "content": (
                            "Evaluate this research topic for a top ML conference paper. "
                            "Score 1-10 on: (a) novelty, (b) specificity, (c) feasibility. "
                            "If overall score < 6, suggest a refined topic.\n\n"
                            f"Topic: {config.research.topic}\n\n"
                            "Reply as JSON: {\"novelty\": N, \"specificity\": N, "
                            "\"feasibility\": N, \"overall\": N, \"suggestion\": \"...\"}"
                        ),
                    }
                ],
                system=(
                    f"You are a senior {_domain_label} "
                    f"researcher evaluating research topic quality."
                ),
            )
            _eval_data = _safe_json_loads(_eval_resp.content, {})
            if isinstance(_eval_data, dict):
                overall = _eval_data.get("overall", 10)
                if isinstance(overall, (int, float)) and overall < 6:
                    # Topic is too broad — treat it as a direction and generate specific candidates
                    logger.warning(
                        "IMP-35: Topic too broad (score %s/10). Generating specific sub-topics...",
                        overall,
                    )
                    try:
                        _refine_resp = llm.chat(
                            [
                                {
                                    "role": "user",
                                    "content": (
                                        f"The research direction '{config.research.topic}' is too broad "
                                        "for a single conference paper. Generate 5 specific, publishable "
                                        "research topics derived from this direction. Each must be concrete "
                                        "enough for a single paper at a top ML/systems venue — specify the "
                                        "mechanism, target system, and approach.\n\n"
                                        "Reply as JSON: {\"candidates\": ["
                                        "{\"topic\": \"...\", \"novelty\": N, \"specificity\": N, "
                                        "\"feasibility\": N, \"overall\": N, \"rationale\": \"...\"}]}"
                                    ),
                                }
                            ],
                            system=(
                                f"You are a senior {_domain_label} researcher helping scope a "
                                "vague research direction into a publishable conference paper topic."
                            ),
                        )
                        _refine_data = _safe_json_loads(_refine_resp.content, {})
                        candidates = _refine_data.get("candidates", [])
                        if candidates:
                            best = max(candidates, key=lambda c: c.get("overall", 0))
                            _eval_data["original_topic"] = config.research.topic
                            _eval_data["refined_topic"] = best["topic"]
                            _eval_data["candidates"] = candidates
                            logger.warning(
                                "IMP-35: Refined topic selected (score %s/10): %s",
                                best.get("overall", "?"),
                                best["topic"],
                            )
                    except Exception:  # noqa: BLE001
                        logger.debug("IMP-35: Sub-topic generation failed (non-blocking)")
                else:
                    logger.info("IMP-35: Topic quality score %s/10", overall)
                (stage_dir / "topic_evaluation.json").write_text(
                    json.dumps(_eval_data, indent=2), encoding="utf-8"
                )
        except Exception:  # noqa: BLE001
            logger.debug("IMP-35: Topic evaluation skipped (non-blocking)")

    return StageResult(
        stage=Stage.PROBLEM_DECOMPOSE,
        status=StageStatus.DONE,
        artifacts=("problem_tree.md",),
        evidence_refs=("stage-02/problem_tree.md",),
    )
