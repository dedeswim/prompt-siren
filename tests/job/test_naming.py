# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for job naming utilities."""

from datetime import datetime, timedelta

from prompt_siren.job.naming import (
    generate_job_name,
    parse_job_name,
    sanitize_for_filename,
)


class TestSanitizeForFilename:
    """Tests for sanitize_for_filename function."""

    def test_replaces_problematic_characters(self):
        """Test that colons, slashes, and spaces are replaced."""
        assert sanitize_for_filename("plain:gpt-5") == "plain_gpt-5"
        assert sanitize_for_filename("azure/gpt-5-turbo") == "azure_gpt-5-turbo"
        assert sanitize_for_filename("my model name") == "my_model_name"

    def test_collapses_multiple_underscores(self):
        """Test that multiple consecutive underscores are collapsed."""
        assert sanitize_for_filename("a::b//c  d") == "a_b_c_d"

    def test_strips_leading_trailing_underscores(self):
        """Test that leading and trailing underscores are removed."""
        assert sanitize_for_filename(":leading") == "leading"
        assert sanitize_for_filename("trailing:") == "trailing"

    def test_handles_special_characters(self):
        """Test that special characters are replaced."""
        assert sanitize_for_filename("model@v1.2.3") == "model_v1_2_3"

    def test_complex_model_names(self):
        """Test sanitization of realistic model names."""
        assert sanitize_for_filename("bedrock:anthropic.claude-3") == "bedrock_anthropic_claude-3"
        assert sanitize_for_filename("openai/gpt-4-turbo-2024") == "openai_gpt-4-turbo-2024"


class TestGenerateJobName:
    """Tests for generate_job_name function."""

    def test_generates_name_with_attack(self):
        """Test generating job name with attack type."""
        timestamp = datetime(2025, 1, 15, 14, 30, 0)
        name = generate_job_name(
            dataset_type="agentdojo-workspace",
            agent_name="plain:gpt-5",
            attack_type="template_string",
            timestamp=timestamp,
        )
        assert name == "agentdojo-workspace_plain_gpt-5_template_string_2025-01-15_14-30-00"

    def test_generates_name_without_attack_uses_benign(self):
        """Test generating job name for benign runs uses 'benign' placeholder."""
        timestamp = datetime(2025, 1, 15, 14, 30, 0)
        name = generate_job_name(
            dataset_type="agentdojo-workspace",
            agent_name="plain:gpt-5",
            attack_type=None,
            timestamp=timestamp,
        )
        assert name == "agentdojo-workspace_plain_gpt-5_benign_2025-01-15_14-30-00"

    def test_sanitizes_all_components(self):
        """Test that all components are sanitized."""
        timestamp = datetime(2025, 1, 15, 14, 30, 0)
        name = generate_job_name(
            dataset_type="dataset:with:colons",
            agent_name="bedrock:anthropic/claude-3",
            attack_type="attack:type",
            timestamp=timestamp,
        )
        assert "dataset_with_colons" in name
        assert "bedrock_anthropic_claude-3" in name
        assert "attack_type" in name

    def test_uses_current_time_when_no_timestamp(self):
        """Test that current time is used when timestamp not provided."""
        before = datetime.now().replace(microsecond=0)
        name = generate_job_name(
            dataset_type="dataset",
            agent_name="agent",
            attack_type=None,
        )
        after = datetime.now().replace(microsecond=0)

        parsed = parse_job_name(name)
        assert parsed is not None
        timestamp = datetime.strptime(parsed["timestamp"], "%Y-%m-%d_%H-%M-%S")
        # Allow 1 second tolerance for edge cases
        assert before - timedelta(seconds=1) <= timestamp <= after + timedelta(seconds=1)


class TestParseJobName:
    """Tests for parse_job_name function."""

    def test_extracts_timestamp_and_prefix(self):
        """Test parsing extracts timestamp and prefix correctly."""
        result = parse_job_name("agentdojo-workspace_plain_gpt-5_template_string_2025-01-15_14-30-00")
        assert result is not None
        assert result["timestamp"] == "2025-01-15_14-30-00"
        assert result["prefix"] == "agentdojo-workspace_plain_gpt-5_template_string"

    def test_returns_none_for_invalid_names(self):
        """Test that invalid job names return None."""
        assert parse_job_name("no_timestamp_here") is None
        assert parse_job_name("prefix_2025-01-15") is None  # incomplete timestamp
        assert parse_job_name("2025-01-15_14-30-00_suffix") is None  # timestamp not at end

    def test_roundtrip_with_generate_job_name(self):
        """Test that generated job names can be parsed back."""
        timestamp = datetime(2025, 6, 15, 10, 30, 45)
        name = generate_job_name(
            dataset_type="test-dataset",
            agent_name="test-agent",
            attack_type="test-attack",
            timestamp=timestamp,
        )
        result = parse_job_name(name)
        assert result is not None
        assert result["timestamp"] == "2025-06-15_10-30-45"
