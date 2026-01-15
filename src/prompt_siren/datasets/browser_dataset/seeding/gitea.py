# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Gitea seeding script.

Seeds a Gitea instance with realistic test data containing injection vector placeholders.
Uses the Gitea API to create users, repositories, issues, and comments.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass, field
from importlib.resources import files

import aiohttp

from .models import GiteaSeedData

logger = logging.getLogger(__name__)


def _load_seed_data() -> GiteaSeedData:
    """Load and validate seed data from JSON file."""
    data_file = (
        files("prompt_siren.datasets.browser_dataset.seeding")
        .joinpath("data")
        .joinpath("gitea.json")
    )
    return GiteaSeedData.model_validate_json(data_file.read_text())


@dataclass
class GiteaSeeder:
    """Seeds a Gitea instance with test data."""

    base_url: str
    admin_username: str = "admin"
    admin_password: str = "admin123"
    _token: str | None = field(default=None, repr=False)

    async def _get_token(self, session: aiohttp.ClientSession) -> str:
        """Get or create an API token."""
        if self._token:
            return self._token

        auth = aiohttp.BasicAuth(self.admin_username, self.admin_password)
        async with session.post(
            f"{self.base_url}/api/v1/users/{self.admin_username}/tokens",
            auth=auth,
            json={"name": "seeding-token", "scopes": ["all"]},
        ) as resp:
            if resp.status == 201:
                data = await resp.json()
                self._token = data["sha1"]
            elif resp.status == 422:
                # Token already exists, delete and recreate
                async with session.delete(
                    f"{self.base_url}/api/v1/users/{self.admin_username}/tokens/seeding-token",
                    auth=auth,
                ) as delete_resp:
                    if delete_resp.status not in (200, 204, 404):
                        logger.warning(
                            "Unexpected status %d when deleting existing token, recreating anyway",
                            delete_resp.status,
                        )
                return await self._get_token(session)
            else:
                raise RuntimeError(f"Failed to create token: {resp.status}")

        if self._token is None:
            raise RuntimeError("Token was not set after successful creation")
        return self._token

    def _headers(self) -> dict[str, str]:
        """Get authorization headers."""
        return {"Authorization": f"token {self._token}"}

    async def create_user(
        self,
        session: aiohttp.ClientSession,
        username: str,
        email: str,
        password: str = "password123",
        full_name: str = "",
    ) -> dict:
        """Create a new user."""
        async with session.post(
            f"{self.base_url}/api/v1/admin/users",
            headers=self._headers(),
            json={
                "username": username,
                "email": email,
                "password": password,
                "full_name": full_name,
                "must_change_password": False,
            },
        ) as resp:
            if resp.status == 201:
                return await resp.json()
            if resp.status == 422:
                # 422 typically means user already exists
                logger.debug(
                    "User %s creation returned 422, assuming user already exists", username
                )
                return {"username": username}
            text = await resp.text()
            raise RuntimeError(f"Failed to create user {username}: {resp.status} {text}")

    async def create_repo(
        self,
        session: aiohttp.ClientSession,
        name: str,
        description: str,
    ) -> dict:
        """Create a new repository."""
        async with session.post(
            f"{self.base_url}/api/v1/user/repos",
            headers=self._headers(),
            json={
                "name": name,
                "description": description,
                "auto_init": True,
                "default_branch": "main",
            },
        ) as resp:
            if resp.status == 201:
                return await resp.json()
            if resp.status == 409:
                async with session.get(
                    f"{self.base_url}/api/v1/repos/{self.admin_username}/{name}",
                    headers=self._headers(),
                ) as get_resp:
                    return await get_resp.json()
            text = await resp.text()
            raise RuntimeError(f"Failed to create repo {name}: {resp.status} {text}")

    async def update_file(
        self,
        session: aiohttp.ClientSession,
        repo_full_name: str,
        path: str,
        content: str,
        message: str,
    ) -> None:
        """Update or create a file in a repository."""
        sha = None
        async with session.get(
            f"{self.base_url}/api/v1/repos/{repo_full_name}/contents/{path}",
            headers=self._headers(),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                sha = data.get("sha")

        payload: dict = {
            "content": base64.b64encode(content.encode()).decode(),
            "message": message,
        }
        if sha:
            payload["sha"] = sha

        async with session.put(
            f"{self.base_url}/api/v1/repos/{repo_full_name}/contents/{path}",
            headers=self._headers(),
            json=payload,
        ) as resp:
            if resp.status not in (200, 201):
                text = await resp.text()
                raise RuntimeError(f"Failed to update file {path}: {resp.status} {text}")

    async def create_issue(
        self,
        session: aiohttp.ClientSession,
        repo_full_name: str,
        title: str,
        body: str,
    ) -> dict:
        """Create an issue."""
        async with session.post(
            f"{self.base_url}/api/v1/repos/{repo_full_name}/issues",
            headers=self._headers(),
            json={"title": title, "body": body},
        ) as resp:
            if resp.status == 201:
                return await resp.json()
            text = await resp.text()
            raise RuntimeError(f"Failed to create issue: {resp.status} {text}")

    async def create_issue_comment(
        self,
        session: aiohttp.ClientSession,
        repo_full_name: str,
        issue_number: int,
        body: str,
    ) -> dict:
        """Create an issue comment."""
        async with session.post(
            f"{self.base_url}/api/v1/repos/{repo_full_name}/issues/{issue_number}/comments",
            headers=self._headers(),
            json={"body": body},
        ) as resp:
            if resp.status == 201:
                return await resp.json()
            text = await resp.text()
            raise RuntimeError(f"Failed to create comment: {resp.status} {text}")

    async def seed(self, session: aiohttp.ClientSession) -> None:
        """Seed the Gitea instance with test data."""
        await self._get_token(session)

        # Load and validate seed data
        data = _load_seed_data()

        # Create users
        for user in data.users:
            await self.create_user(
                session,
                username=user.username,
                email=user.email,
                full_name=user.full_name,
            )

        # Create repositories with files
        repos: dict[str, dict] = {}
        for repo_def in data.repositories:
            repo = await self.create_repo(
                session,
                name=repo_def.name,
                description=repo_def.description,
            )
            repos[repo_def.name] = repo

            # Add files
            for file_path, content in repo_def.files.items():
                await self.update_file(
                    session,
                    repo["full_name"],
                    file_path,
                    content,
                    f"Add {file_path}",
                )

        # Create issues with comments
        for issue_def in data.issues:
            repo = repos[issue_def.repo]
            issue = await self.create_issue(
                session,
                repo["full_name"],
                issue_def.title,
                issue_def.body,
            )

            for comment in issue_def.comments:
                await self.create_issue_comment(
                    session,
                    repo["full_name"],
                    issue["number"],
                    comment,
                )


async def seed_gitea(
    base_url: str = "http://gitea.dev-forge.io",
    admin_username: str = "admin",
    admin_password: str = "admin123",
) -> None:
    """Seed a Gitea instance with test data.

    Args:
        base_url: Base URL of the Gitea instance
        admin_username: Admin username for API access
        admin_password: Admin password for API access
    """
    seeder = GiteaSeeder(
        base_url=base_url,
        admin_username=admin_username,
        admin_password=admin_password,
    )

    async with aiohttp.ClientSession() as session:
        await seeder.seed(session)


if __name__ == "__main__":
    asyncio.run(seed_gitea())
