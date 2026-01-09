# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Base class for browser datasets with different observation modalities.

This module provides a base class that handles common browser dataset
functionality (container setup, task definitions, injection handling) while
allowing subclasses to customize observation rendering and tools.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from playwright.async_api import Page
from pydantic_ai.toolsets import FunctionToolset

from ...environments.abstract import AbstractEnvironment
from ...environments.browser_env import (
    apply_injections,
    BrowserEnvironment,
    BrowserEnvState,
    BrowserTaskMetadata,
)
from ...sandbox_managers.abstract import AbstractSandboxManager
from ...tasks import BenignTask, MaliciousTask, TaskCouple
from ...types import InjectionAttacksDict, StrContentAttack
from ..abstract import AbstractDataset
from .config import BrowserDatasetConfig, SiteName
from .injection import get_vectors_for_sites
from .malicious_tasks import MALICIOUS_TASKS
from .sites import ANSWER_BENIGN_TASKS, GITEA_BENIGN_TASKS

# Output type varies by observation modality
OutputT = TypeVar("OutputT")

# All tasks (flat lists)
ALL_BENIGN_TASKS: list[BenignTask[BrowserEnvState]] = GITEA_BENIGN_TASKS + ANSWER_BENIGN_TASKS
ALL_MALICIOUS_TASKS: list[MaliciousTask[BrowserEnvState]] = MALICIOUS_TASKS


def _compute_sites_with_tasks() -> frozenset[SiteName]:
    """Compute all sites that have tasks defined (called once at module load)."""
    sites: set[SiteName] = set()

    for task in ALL_BENIGN_TASKS + ALL_MALICIOUS_TASKS:
        if isinstance(task.metadata, BrowserTaskMetadata):
            sites.update(task.metadata.sites)

    return frozenset(sites)


# Computed once at module load
SITES_WITH_TASKS: frozenset[SiteName] = _compute_sites_with_tasks()


# Type alias for render function
RenderFn = Callable[[Page, InjectionAttacksDict[StrContentAttack] | None], Awaitable[OutputT]]


@dataclass(frozen=True)
class BaseBrowserDataset(
    AbstractDataset[BrowserEnvState, Any, OutputT, StrContentAttack],
    Generic[OutputT],
):
    """Base class for browser datasets with different observation modalities.

    This class handles shared functionality:
    - Container setup and management
    - Task definitions (benign, malicious, couples)
    - Injection handling

    Concrete dataset classes are created via factory functions that configure:
    - Observation rendering (screenshot, a11y tree, HTML)
    - Tool definitions appropriate for the observation type
    - System prompts guiding the agent
    """

    name: str
    _environment: BrowserEnvironment[OutputT]
    _benign_tasks: list[BenignTask[BrowserEnvState]] = field(default_factory=list)
    _malicious_tasks: list[MaliciousTask[BrowserEnvState]] = field(default_factory=list)
    _task_couples: list[TaskCouple[BrowserEnvState]] = field(default_factory=list)
    _toolsets: list[FunctionToolset[BrowserEnvState]] = field(default_factory=list)
    _system_prompt: str | None = None

    @property
    def system_prompt(self) -> str | None:
        return self._system_prompt

    @property
    def environment(
        self,
    ) -> AbstractEnvironment[BrowserEnvState, Any, OutputT, StrContentAttack]:
        return self._environment

    @property
    def default_toolsets(self) -> list[FunctionToolset[BrowserEnvState]]:
        return self._toolsets

    @property
    def benign_tasks(self) -> list[BenignTask[BrowserEnvState]]:
        return self._benign_tasks

    @property
    def malicious_tasks(self) -> list[MaliciousTask[BrowserEnvState]]:
        return self._malicious_tasks

    @property
    def task_couples(self) -> list[TaskCouple[BrowserEnvState]]:
        return self._task_couples


def create_browser_environment(
    config: BrowserDatasetConfig,
    sandbox_manager: AbstractSandboxManager,
    render_fn: RenderFn[OutputT],
    *,
    name: str = "browser",
) -> BrowserEnvironment[OutputT]:
    """Create a browser environment with the given render function.

    Args:
        config: Browser dataset configuration
        sandbox_manager: Sandbox manager for container lifecycle
        render_fn: Function to render Page to observation format
        name: Name identifier for the environment

    Returns:
        Configured BrowserEnvironment
    """
    # Use pre-computed sites from module load
    sites = SITES_WITH_TASKS

    # Build site URL map
    site_urls: dict[str, str] = {}
    for site_name in sites:
        site_config = config.get_site_config(site_name)
        site_urls[site_name] = site_config.get_url()

    return BrowserEnvironment(
        name=name,
        all_injection_ids=get_vectors_for_sites(list(sites)),
        sandbox_manager=sandbox_manager,
        browser_container_spec=config.browser.to_container_spec(),
        site_container_specs=config.get_all_site_container_specs(),
        site_urls=site_urls,
        render_fn=render_fn,
    )
