# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for the build_images script."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

# Skip this entire module if swebench is not installed
pytest.importorskip("swebench")

from prompt_siren.build_images import build_dataset_images, ImageBuilder
from prompt_siren.datasets.swebench_dataset.image_tags import (
    get_benign_image_tag,
    get_pair_image_tag,
)


class TestImageTagFunctions:
    """Test image tag generation functions."""

    def test_get_benign_image_tag(self) -> None:
        """Test benign image tag generation."""
        tag = get_benign_image_tag("django__django-11179")
        assert tag == "siren-swebench-benign:django__django-11179"

    def test_get_pair_image_tag(self) -> None:
        """Test pair image tag generation."""
        tag = get_pair_image_tag("django__django-11179", "env_exfil_task")
        assert tag == "siren-swebench-pair:django__django-11179__env_exfil_task"


class MockDockerClient:
    """Mock Docker client for testing."""

    def __init__(self) -> None:
        self.inspect_image = AsyncMock()
        self.delete_image = AsyncMock()
        self.tag_image = AsyncMock()
        self.push_image = AsyncMock()


class TestBuildDatasetImagesValidation:
    """Tests for build_dataset_images config validation."""

    @pytest.fixture
    def mock_docker(self) -> MockDockerClient:
        return MockDockerClient()

    @pytest.fixture
    def builder(self, mock_docker: MockDockerClient, tmp_path: Path) -> ImageBuilder:
        return ImageBuilder(
            docker_client=mock_docker,  # type: ignore[arg-type]
            cache_dir=tmp_path,
        )

    @pytest.mark.anyio
    async def test_raises_value_error_on_invalid_config(self, builder: ImageBuilder) -> None:
        """Verify ValueError when config overrides are invalid."""
        with pytest.raises(ValueError, match="Invalid configuration"):
            await build_dataset_images(
                "swebench",
                builder,
                max_instances="not-an-integer",  # Invalid: should be int
            )
