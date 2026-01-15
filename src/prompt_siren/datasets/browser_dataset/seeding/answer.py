# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Apache Answer seeding script.

Seeds an Apache Answer instance with realistic Q&A test data containing
injection vector placeholders.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from importlib.resources import files

import aiohttp

from .models import AnswerSeedData

logger = logging.getLogger(__name__)


def _load_seed_data() -> AnswerSeedData:
    """Load and validate seed data from JSON file."""
    data_file = (
        files("prompt_siren.datasets.browser_dataset.seeding")
        .joinpath("data")
        .joinpath("answer.json")
    )
    return AnswerSeedData.model_validate_json(data_file.read_text())


@dataclass
class AnswerSeeder:
    """Seeds an Apache Answer instance with test data."""

    base_url: str
    admin_username: str = "admin"
    admin_password: str = "admin123"
    _token: str | None = field(default=None, repr=False)

    async def _login(self, session: aiohttp.ClientSession) -> str:
        """Login and get access token."""
        if self._token:
            return self._token

        async with session.post(
            f"{self.base_url}/answer/api/v1/user/login/email",
            json={
                "e_mail": f"{self.admin_username}@example.com",
                "pass": self.admin_password,
            },
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                self._token = data.get("data", {}).get("access_token")
                if not self._token:
                    raise RuntimeError(f"No access token in response: {data}")
            else:
                text = await resp.text()
                raise RuntimeError(f"Failed to login: {resp.status} {text}")

        return self._token

    def _headers(self) -> dict[str, str]:
        """Get authorization headers."""
        return {"Authorization": f"Bearer {self._token}"}

    async def create_question(
        self,
        session: aiohttp.ClientSession,
        title: str,
        content: str,
        tags: list[str] | None = None,
    ) -> dict:
        """Create a new question."""
        payload = {
            "title": title,
            "content": content,
            "tags": [{"slug_name": tag} for tag in (tags or [])],
        }

        async with session.post(
            f"{self.base_url}/answer/api/v1/question",
            headers=self._headers(),
            json=payload,
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            text = await resp.text()
            raise RuntimeError(f"Failed to create question: {resp.status} {text}")

    async def create_answer(
        self,
        session: aiohttp.ClientSession,
        question_id: str,
        content: str,
    ) -> dict:
        """Create an answer to a question."""
        async with session.post(
            f"{self.base_url}/answer/api/v1/answer",
            headers=self._headers(),
            json={
                "question_id": question_id,
                "content": content,
            },
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            text = await resp.text()
            raise RuntimeError(f"Failed to create answer: {resp.status} {text}")

    async def create_comment(
        self,
        session: aiohttp.ClientSession,
        object_id: str,
        content: str,
    ) -> dict:
        """Create a comment on a question or answer."""
        async with session.post(
            f"{self.base_url}/answer/api/v1/comment",
            headers=self._headers(),
            json={
                "object_id": object_id,
                "original_text": content,
            },
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            text = await resp.text()
            raise RuntimeError(f"Failed to create comment: {resp.status} {text}")

    async def update_user_bio(
        self,
        session: aiohttp.ClientSession,
        bio: str,
    ) -> dict:
        """Update the current user's bio."""
        async with session.put(
            f"{self.base_url}/answer/api/v1/user/info",
            headers=self._headers(),
            json={"bio": bio},
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            text = await resp.text()
            raise RuntimeError(f"Failed to update bio: {resp.status} {text}")

    async def seed(self, session: aiohttp.ClientSession) -> None:
        """Seed the Answer instance with test data."""
        await self._login(session)

        # Load and validate seed data
        data = _load_seed_data()

        # Update user bio with injection vector
        if data.user_bio:
            await self.update_user_bio(session, bio=data.user_bio)

        # Create all questions with their answers and comments
        for q_def in data.questions:
            question = await self.create_question(
                session,
                title=q_def.title,
                content=q_def.content,
                tags=q_def.tags,
            )

            q_id = question.get("data", {}).get("id", "")
            if not q_id:
                logger.warning(
                    "Failed to get question ID for '%s', skipping answers and comments. Response: %s",
                    q_def.title,
                    question,
                )
                continue

            for answer_def in q_def.answers:
                answer = await self.create_answer(
                    session,
                    question_id=q_id,
                    content=answer_def.content,
                )

                a_id = answer.get("data", {}).get("id", "")
                if not a_id:
                    logger.warning(
                        "Failed to get answer ID for question '%s', skipping comments. Response: %s",
                        q_def.title,
                        answer,
                    )
                    continue

                for comment in answer_def.comments:
                    await self.create_comment(
                        session,
                        object_id=a_id,
                        content=comment,
                    )


async def seed_answer(
    base_url: str = "http://answers.dev-community.io",
    admin_username: str = "admin",
    admin_password: str = "admin123",
) -> None:
    """Seed an Apache Answer instance with test data.

    Args:
        base_url: Base URL of the Answer instance
        admin_username: Admin username for API access
        admin_password: Admin password for API access
    """
    seeder = AnswerSeeder(
        base_url=base_url,
        admin_username=admin_username,
        admin_password=admin_password,
    )

    async with aiohttp.ClientSession() as session:
        await seeder.seed(session)


if __name__ == "__main__":
    asyncio.run(seed_answer())
