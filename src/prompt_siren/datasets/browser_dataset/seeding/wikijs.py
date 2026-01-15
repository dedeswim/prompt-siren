# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Wiki.js seeding script.

Seeds a Wiki.js instance with documentation pages containing injection vector placeholders.
Uses the Wiki.js GraphQL API to create pages.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from importlib.resources import files

import aiohttp

from .models import WikiSeedData

# GraphQL mutations for Wiki.js
LOGIN_MUTATION = """
mutation Login($username: String!, $password: String!, $strategy: String!) {
  authentication {
    login(username: $username, password: $password, strategy: $strategy) {
      responseResult {
        succeeded
        errorCode
        message
      }
      jwt
    }
  }
}
"""

CREATE_PAGE_MUTATION = """
mutation CreatePage($content: String!, $description: String!, $editor: String!, $isPublished: Boolean!, $isPrivate: Boolean!, $locale: String!, $path: String!, $tags: [String]!, $title: String!) {
  pages {
    create(content: $content, description: $description, editor: $editor, isPublished: $isPublished, isPrivate: $isPrivate, locale: $locale, path: $path, tags: $tags, title: $title) {
      responseResult {
        succeeded
        errorCode
        message
      }
      page {
        id
        path
        title
      }
    }
  }
}
"""


def _load_seed_data() -> WikiSeedData:
    """Load and validate seed data from JSON file."""
    data_file = (
        files("prompt_siren.datasets.browser_dataset.seeding")
        .joinpath("data")
        .joinpath("wikijs.json")
    )
    return WikiSeedData.model_validate_json(data_file.read_text())


@dataclass
class WikiJsSeeder:
    """Seeds a Wiki.js instance with test data."""

    base_url: str
    admin_email: str = "admin@example.com"
    admin_password: str = "admin123"
    _jwt: str | None = field(default=None, repr=False)

    async def _graphql(
        self,
        session: aiohttp.ClientSession,
        query: str,
        variables: dict | None = None,
    ) -> dict:
        """Execute a GraphQL query."""
        headers = {"Content-Type": "application/json"}
        if self._jwt:
            headers["Authorization"] = f"Bearer {self._jwt}"

        async with session.post(
            f"{self.base_url}/graphql",
            headers=headers,
            json={"query": query, "variables": variables or {}},
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"GraphQL request failed: {resp.status} {text}")
            return await resp.json()

    async def _login(self, session: aiohttp.ClientSession) -> str:
        """Login and get JWT token."""
        if self._jwt:
            return self._jwt

        result = await self._graphql(
            session,
            LOGIN_MUTATION,
            {
                "username": self.admin_email,
                "password": self.admin_password,
                "strategy": "local",
            },
        )

        login_data = result.get("data", {}).get("authentication", {}).get("login", {})
        response_result = login_data.get("responseResult", {})

        if not response_result.get("succeeded"):
            raise RuntimeError(f"Login failed: {response_result.get('message', 'Unknown error')}")

        self._jwt = login_data.get("jwt")
        if not self._jwt:
            raise RuntimeError("No JWT token in login response")

        return self._jwt

    async def create_page(
        self,
        session: aiohttp.ClientSession,
        path: str,
        title: str,
        content: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> dict:
        """Create a wiki page."""
        result = await self._graphql(
            session,
            CREATE_PAGE_MUTATION,
            {
                "content": content,
                "description": description,
                "editor": "markdown",
                "isPublished": True,
                "isPrivate": False,
                "locale": "en",
                "path": path,
                "tags": tags or [],
                "title": title,
            },
        )

        create_data = result.get("data", {}).get("pages", {}).get("create", {})
        response_result = create_data.get("responseResult", {})

        if not response_result.get("succeeded"):
            # Page might already exist, which is okay
            error_code = response_result.get("errorCode")
            if error_code == "PageDuplicateCreate":
                return {"path": path, "title": title, "exists": True}
            raise RuntimeError(
                f"Failed to create page {path}: {response_result.get('message', 'Unknown error')}"
            )

        return create_data.get("page", {})

    async def seed(self, session: aiohttp.ClientSession) -> None:
        """Seed the Wiki.js instance with test data."""
        await self._login(session)

        # Load and validate seed data
        data = _load_seed_data()

        # Create all pages
        for page in data.pages:
            await self.create_page(
                session,
                path=page.path,
                title=page.title,
                content=page.content,
            )


async def seed_wikijs(
    base_url: str = "http://wiki.internal-docs.io",
    admin_email: str = "admin@example.com",
    admin_password: str = "admin123",
) -> None:
    """Seed a Wiki.js instance with test data.

    Args:
        base_url: Base URL of the Wiki.js instance
        admin_email: Admin email for API access
        admin_password: Admin password for API access
    """
    seeder = WikiJsSeeder(
        base_url=base_url,
        admin_email=admin_email,
        admin_password=admin_password,
    )

    async with aiohttp.ClientSession() as session:
        await seeder.seed(session)


if __name__ == "__main__":
    asyncio.run(seed_wikijs())
