# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Pydantic models for seed data structures."""

from __future__ import annotations

from pydantic import BaseModel


# Gitea models
class GiteaUser(BaseModel):
    """A Gitea user to create."""

    username: str
    email: str
    full_name: str = ""


class GiteaRepository(BaseModel):
    """A Gitea repository with files."""

    name: str
    description: str
    files: dict[str, str] = {}


class GiteaIssue(BaseModel):
    """A Gitea issue with comments."""

    repo: str
    title: str
    body: str
    comments: list[str] = []


class GiteaSeedData(BaseModel):
    """Complete Gitea seed data."""

    users: list[GiteaUser] = []
    repositories: list[GiteaRepository] = []
    issues: list[GiteaIssue] = []


# Answer models
class AnswerComment(BaseModel):
    """A comment on an answer."""

    content: str


class AnswerAnswer(BaseModel):
    """An answer to a question."""

    content: str
    comments: list[str] = []


class AnswerQuestion(BaseModel):
    """A question with answers."""

    title: str
    content: str
    tags: list[str] = []
    answers: list[AnswerAnswer] = []


class AnswerSeedData(BaseModel):
    """Complete Answer seed data."""

    user_bio: str = ""
    questions: list[AnswerQuestion] = []


# Wiki.js models
class WikiPage(BaseModel):
    """A Wiki.js page."""

    path: str
    title: str
    content: str


class WikiSeedData(BaseModel):
    """Complete Wiki.js seed data."""

    pages: list[WikiPage] = []
