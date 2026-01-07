# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Configuration for browser-based dataset."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Discriminator, Field, Tag
from typing_extensions import assert_never

from ...environments.browser_env import SiteName
from ...sandbox_managers.image_spec import PullImageSpec
from ...sandbox_managers.sandbox_task_setup import ContainerSpec

# Default browser container image (Headless Chrome with CDP support)
# chromedp/headless-shell is Debian-based and designed for CDP usage
DEFAULT_BROWSER_IMAGE = "chromedp/headless-shell:latest"

# CDP port for browser container
CDP_PORT = 9222


class BaseSiteConfig(BaseModel):
    """Common configuration for all sites."""

    container_image: str
    """Docker image for the site container."""
    hostname: str
    """Realistic hostname for the site (e.g., 'gitea.dev-forge.io').

    This hostname is used in task prompts and URLs. Docker DNS resolves
    this hostname to the container when using containerized browser mode.
    """
    port: int
    """Port the site runs on inside the container."""
    base_url: str | None = None
    """Override base URL for the site. If not set, defaults to http://{hostname}:{port}."""

    def get_url(self) -> str:
        """Get the effective URL for this site.

        Returns base_url if set, otherwise constructs URL from hostname and port.
        Port 80 is omitted from URL as it's the default HTTP port.
        """
        if self.base_url:
            return self.base_url
        if self.port == 80:
            return f"http://{self.hostname}"
        return f"http://{self.hostname}:{self.port}"

    def to_container_spec(self) -> ContainerSpec:
        """Convert site config to ContainerSpec for sandbox manager."""
        return ContainerSpec(
            image_spec=PullImageSpec(tag=self.container_image),
            hostname=self.hostname,
            ports={self.port: self.port},
        )


class SqliteSiteConfig(BaseSiteConfig):
    """Configuration for SQLite-backed sites (easy checkpointing via file copy)."""

    db_type: Literal["sqlite"] = "sqlite"
    """Database type (always sqlite for this config)."""
    db_path: Path
    """Path to the SQLite database file inside the container."""


class PostgresqlSiteConfig(BaseSiteConfig):
    """Configuration for PostgreSQL-backed sites (requires separate DB container)."""

    db_type: Literal["postgresql"] = "postgresql"
    """Database type (always postgresql for this config)."""
    db_image: str
    """Docker image for the PostgreSQL database container."""


SiteConfig = Annotated[
    Annotated[SqliteSiteConfig, Tag("sqlite")] | Annotated[PostgresqlSiteConfig, Tag("postgresql")],
    Discriminator("db_type"),
]
"""Site configuration - either SQLite or PostgreSQL backed."""


class BrowserContainerConfig(BaseModel):
    """Configuration for the browser container (Chromium with CDP)."""

    image: str = DEFAULT_BROWSER_IMAGE
    """Docker image for the browser container."""
    cdp_port: int = CDP_PORT
    """CDP port for remote debugging."""

    def to_container_spec(self) -> ContainerSpec:
        """Convert to ContainerSpec for sandbox manager.

        Note: chromedp/headless-shell has a built-in ENTRYPOINT that runs
        headless Chrome with CDP on port 9222. We don't set command so the
        image's default entrypoint is used.
        """
        return ContainerSpec(
            image_spec=PullImageSpec(tag=self.image),
            hostname="browser",
            ports={self.cdp_port: self.cdp_port},
        )


class BrowserDatasetConfig(BaseModel):
    """Configuration for browser-based dataset.

    This configuration supports multiple website containers (Gitea, Apache Answer,
    Wiki.js, VWA Classifieds) managed by the sandbox manager.

    Container Management:
        Follows the SWE-bench pattern:
        - setup_batch(): Pulls/prepares all container images upfront
        - setup_task(): Creates fresh browser + site containers per task
        - Complete isolation between tasks (no shared state)
        - Supports true parallel execution
    """

    # Browser container configuration
    browser: BrowserContainerConfig = Field(
        default_factory=BrowserContainerConfig,
        description="Browser container configuration.",
    )

    # SQLite sites (lightweight, easy checkpointing)
    gitea: SqliteSiteConfig = Field(
        default=SqliteSiteConfig(
            container_image="gitea/gitea:latest",
            hostname="gitea.dev-forge.io",
            db_path=Path("/data/gitea/gitea.db"),
            port=80,
        ),
        description="Gitea configuration (Git forge with issues, PRs, code review).",
    )

    answer: SqliteSiteConfig = Field(
        default=SqliteSiteConfig(
            container_image="apache/answer:latest",
            hostname="answers.dev-community.io",
            db_path=Path("/data/answer.db"),
            port=80,
        ),
        description="Apache Answer configuration (Q&A platform).",
    )

    wikijs: SqliteSiteConfig = Field(
        default=SqliteSiteConfig(
            container_image="linuxserver/wikijs:latest",
            hostname="wiki.internal-docs.io",
            db_path=Path("/config/database.sqlite"),
            port=80,
        ),
        description="Wiki.js configuration (wiki platform).",
    )

    # PostgreSQL site (heavier, for complex scenarios)
    classifieds: PostgresqlSiteConfig = Field(
        default=PostgresqlSiteConfig(
            container_image="ghcr.io/bgrins/vwa_classifieds_web:1",
            hostname="marketplace.local-listings.io",
            db_image="ghcr.io/bgrins/vwa_classifieds_db:1",
            port=80,
        ),
        description="VWA Classifieds configuration (marketplace with PostgreSQL).",
    )

    def get_site_config(self, site_name: SiteName) -> SiteConfig:
        """Get configuration for a specific site.

        Args:
            site_name: Name of the site

        Returns:
            SiteConfig for the requested site
        """
        match site_name:
            case "gitea":
                return self.gitea
            case "answer":
                return self.answer
            case "wikijs":
                return self.wikijs
            case "classifieds":
                return self.classifieds
            case _:
                assert_never(site_name)

    def get_all_site_container_specs(self) -> dict[str, ContainerSpec]:
        """Get container specs for all sites.

        Returns:
            Dictionary mapping site names to their ContainerSpecs
        """
        return {
            "gitea": self.gitea.to_container_spec(),
            "answer": self.answer.to_container_spec(),
            "wikijs": self.wikijs.to_container_spec(),
            "classifieds": self.classifieds.to_container_spec(),
        }
