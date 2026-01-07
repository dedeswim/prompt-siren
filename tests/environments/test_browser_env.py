# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Unit tests for BrowserEnvironment."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from prompt_siren.environments.browser_env import (
    BrowserEnvironment,
    BrowserTaskMetadata,
)
from prompt_siren.sandbox_managers.image_spec import PullImageSpec
from prompt_siren.sandbox_managers.sandbox_task_setup import ContainerSpec
from prompt_siren.tasks import BenignTask, MaliciousTask, TaskCouple

pytestmark = pytest.mark.anyio


@pytest.fixture
def mock_sandbox_manager() -> MagicMock:
    """Create a mock sandbox manager for testing."""
    manager = MagicMock()
    manager.clone = AsyncMock()
    manager.setup_batch = AsyncMock()
    manager.setup_task = AsyncMock()
    return manager


@pytest.fixture
def browser_container_spec() -> ContainerSpec:
    """Create browser container spec."""
    return ContainerSpec(
        image_spec=PullImageSpec(tag="chromedp/headless-shell:latest"),
        hostname="browser",
        ports={9222: 9222},
    )


@pytest.fixture
def site_container_specs() -> dict[str, ContainerSpec]:
    """Create site container specs."""
    return {
        "gitea": ContainerSpec(
            image_spec=PullImageSpec(tag="gitea/gitea:latest"),
            hostname="gitea.dev-forge.io",
            ports={80: 80},
        ),
        "answer": ContainerSpec(
            image_spec=PullImageSpec(tag="apache/answer:latest"),
            hostname="answers.dev-community.io",
            ports={80: 80},
        ),
    }


@pytest.fixture
def browser_env(
    mock_sandbox_manager: MagicMock,
    browser_container_spec: ContainerSpec,
    site_container_specs: dict[str, ContainerSpec],
) -> BrowserEnvironment:
    """Create a BrowserEnvironment instance for testing."""
    injection_ids = ["gitea_issue_content", "gitea_readme", "answer_question_body"]
    return BrowserEnvironment(
        name="test-browser",
        all_injection_ids=injection_ids,
        sandbox_manager=mock_sandbox_manager,
        browser_container_spec=browser_container_spec,
        site_container_specs=site_container_specs,
        site_urls={
            "gitea": "http://gitea.dev-forge.io",
            "answer": "http://answers.dev-community.io",
        },
    )


class TestGetInjectableIds:
    """Tests for get_injectable_ids method."""

    async def test_finds_single_injection_id(self, browser_env: BrowserEnvironment):
        """Test finding a single injection ID in page content."""
        mock_page = MagicMock()
        mock_page.content = AsyncMock(
            return_value="<html><body>Issue: {gitea_issue_content}</body></html>"
        )

        result = await browser_env.get_injectable_ids(mock_page)

        assert "gitea_issue_content" in result
        assert len(result) == 1

    async def test_finds_multiple_injection_ids(self, browser_env: BrowserEnvironment):
        """Test finding multiple injection IDs in page content."""
        mock_page = MagicMock()
        mock_page.content = AsyncMock(
            return_value="<html><body>{gitea_issue_content} and {gitea_readme}</body></html>"
        )

        result = await browser_env.get_injectable_ids(mock_page)

        assert "gitea_issue_content" in result
        assert "gitea_readme" in result
        assert len(result) == 2

    async def test_does_not_match_partial_braces(self, browser_env: BrowserEnvironment):
        """Test that incomplete braces don't match."""
        mock_page = MagicMock()
        mock_page.content = AsyncMock(
            return_value="<html><body>gitea_issue_content without braces</body></html>"
        )

        result = await browser_env.get_injectable_ids(mock_page)

        assert result == []

    async def test_only_matches_known_injection_ids(self, browser_env: BrowserEnvironment):
        """Test that only IDs in all_injection_ids are matched, not arbitrary {text}."""
        mock_page = MagicMock()
        mock_page.content = AsyncMock(
            return_value="<html><body>{unknown_vector} and {gitea_readme}</body></html>"
        )

        result = await browser_env.get_injectable_ids(mock_page)

        # Should only find gitea_readme (in all_injection_ids), not unknown_vector
        assert result == ["gitea_readme"]


class TestGetSitesFromTask:
    """Tests for _get_sites_from_task method."""

    def test_single_site_from_benign_task(self, browser_env: BrowserEnvironment):
        """Test extracting single site from BenignTask."""
        task = BenignTask(
            id="test_task",
            prompt="Do something",
            evaluators={},
            metadata=BrowserTaskMetadata(sites=["gitea"]),
        )

        result = browser_env._get_sites_from_task(task)

        assert result == ["gitea"]

    def test_multiple_sites_from_cross_site_task(self, browser_env: BrowserEnvironment):
        """Test extracting multiple sites from BrowserTaskMetadata."""
        task = BenignTask(
            id="cross_site_task",
            prompt="Do something across sites",
            evaluators={},
            metadata=BrowserTaskMetadata(sites=["gitea", "answer"]),
        )

        result = browser_env._get_sites_from_task(task)

        # Order is preserved from metadata
        assert result == ["gitea", "answer"]

    def test_combines_sites_from_task_couple(self, browser_env: BrowserEnvironment):
        """Test that TaskCouple combines sites from both benign and malicious tasks."""
        benign = BenignTask(
            id="benign_task",
            prompt="Do something",
            evaluators={},
            metadata=BrowserTaskMetadata(sites=["gitea"]),
        )
        malicious = MaliciousTask(
            id="malicious_task",
            goal="Attack",
            evaluators={},
            metadata=BrowserTaskMetadata(sites=["answer"]),
        )
        couple = TaskCouple(benign=benign, malicious=malicious)

        result = browser_env._get_sites_from_task(couple)

        # Benign sites come first, then malicious (order preserved)
        assert result == ["gitea", "answer"]

    def test_combines_cross_site_with_single_site(self, browser_env: BrowserEnvironment):
        """Test combining BrowserTaskMetadata with single-site task in couple."""
        benign = BenignTask(
            id="benign_task",
            prompt="Do something",
            evaluators={},
            metadata=BrowserTaskMetadata(sites=["gitea"]),
        )
        malicious = MaliciousTask(
            id="malicious_task",
            goal="Attack across sites",
            evaluators={},
            metadata=BrowserTaskMetadata(sites=["answer", "wikijs"]),
        )
        couple = TaskCouple(benign=benign, malicious=malicious)

        result = browser_env._get_sites_from_task(couple)

        # Benign first, then malicious (order preserved, no duplicates)
        assert result == ["gitea", "answer", "wikijs"]

    def test_deduplicates_same_site(self, browser_env: BrowserEnvironment):
        """Test that same site in both tasks is deduplicated."""
        benign = BenignTask(
            id="benign_task",
            prompt="Do something",
            evaluators={},
            metadata=BrowserTaskMetadata(sites=["gitea"]),
        )
        malicious = MaliciousTask(
            id="malicious_task",
            goal="Attack",
            evaluators={},
            metadata=BrowserTaskMetadata(sites=["gitea"]),
        )
        couple = TaskCouple(benign=benign, malicious=malicious)

        result = browser_env._get_sites_from_task(couple)

        # Should deduplicate
        assert result == ["gitea"]

    def test_include_malicious_false_returns_only_benign_sites(
        self, browser_env: BrowserEnvironment
    ):
        """Test that include_malicious=False excludes malicious task sites."""
        benign = BenignTask(
            id="benign_task",
            prompt="Do something",
            evaluators={},
            metadata=BrowserTaskMetadata(sites=["gitea"]),
        )
        malicious = MaliciousTask(
            id="malicious_task",
            goal="Attack",
            evaluators={},
            metadata=BrowserTaskMetadata(sites=["answer"]),
        )
        couple = TaskCouple(benign=benign, malicious=malicious)

        result = browser_env._get_sites_from_task(couple, include_malicious=False)

        # Should only return benign task's sites
        assert result == ["gitea"]

    def test_extracts_first_site_for_url_resolution(self, browser_env: BrowserEnvironment):
        """Test that first site can be used for URL resolution."""
        task = BenignTask(
            id="cross_site_task",
            prompt="Do something across sites",
            evaluators={},
            metadata=BrowserTaskMetadata(sites=["answer", "gitea"]),
        )

        result = browser_env._get_sites_from_task(task)

        # First element is primary site for URL resolution
        assert result[0] == "answer"


class TestCreateTaskSetup:
    """Tests for _create_task_setup method."""

    def test_creates_setup_for_single_site_task(self, browser_env: BrowserEnvironment):
        """Test creating TaskSetup for a single-site benign task."""
        task = BenignTask(
            id="gitea_find_issue",
            prompt="Find the issue",
            evaluators={},
            metadata=BrowserTaskMetadata(sites=["gitea"]),
        )

        setup = browser_env._create_task_setup(task)

        assert setup.task_id == "gitea_find_issue"
        assert setup.agent_container.name == "browser"
        assert "gitea" in setup.service_containers
        assert setup.network_config is not None
        assert setup.network_config.name == "browser-net-gitea_find_issue"
        assert setup.network_config.internal is False

    def test_creates_setup_for_cross_site_task(self, browser_env: BrowserEnvironment):
        """Test creating TaskSetup for a cross-site task."""
        task = BenignTask(
            id="cross_site_task",
            prompt="Do something across sites",
            evaluators={},
            metadata=BrowserTaskMetadata(sites=["gitea", "answer"]),
        )

        setup = browser_env._create_task_setup(task)

        assert setup.task_id == "cross_site_task"
        assert "gitea" in setup.service_containers
        assert "answer" in setup.service_containers
        assert len(setup.service_containers) == 2

    def test_creates_setup_for_task_couple(self, browser_env: BrowserEnvironment):
        """Test creating TaskSetup for a TaskCouple."""
        benign = BenignTask(
            id="benign_task",
            prompt="Do something",
            evaluators={},
            metadata=BrowserTaskMetadata(sites=["gitea"]),
        )
        malicious = MaliciousTask(
            id="malicious_task",
            goal="Attack",
            evaluators={},
            metadata=BrowserTaskMetadata(sites=["answer"]),
        )
        couple = TaskCouple(benign=benign, malicious=malicious)

        setup = browser_env._create_task_setup(couple)

        # Couple ID format is "benign_id:malicious_id"
        assert setup.task_id == "benign_task:malicious_task"
        # Should include containers for both sites
        assert "gitea" in setup.service_containers
        assert "answer" in setup.service_containers

    def test_sanitizes_task_id_for_network_name(self, browser_env: BrowserEnvironment):
        """Test that task IDs with special characters are sanitized for network names."""
        benign = BenignTask(
            id="benign/task",
            prompt="Do something",
            evaluators={},
            metadata=BrowserTaskMetadata(sites=["gitea"]),
        )
        malicious = MaliciousTask(
            id="malicious:task",
            goal="Attack",
            evaluators={},
            metadata=BrowserTaskMetadata(sites=["gitea"]),
        )
        couple = TaskCouple(benign=benign, malicious=malicious)

        setup = browser_env._create_task_setup(couple)

        # Colons and slashes should be replaced with dashes
        assert setup.network_config is not None
        assert ":" not in setup.network_config.name
        assert "/" not in setup.network_config.name
        assert setup.network_config.name == "browser-net-benign-task-malicious-task"

    def test_skips_unknown_site_containers(self, browser_env: BrowserEnvironment):
        """Test that unknown sites don't cause container creation errors."""
        # wikijs is not in our site_container_specs fixture
        task = BenignTask(
            id="wiki_task",
            prompt="Do something on wiki",
            evaluators={},
            metadata=BrowserTaskMetadata(sites=["gitea", "wikijs"]),
        )

        setup = browser_env._create_task_setup(task)

        # Only gitea should be in service containers (wikijs not configured)
        assert "gitea" in setup.service_containers
        assert "wikijs" not in setup.service_containers
