# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Site data seeding for browser dataset.

This module provides utilities to pre-populate site containers with test data
containing `{vector_id}` placeholders that the injection system can detect
and replace with attack content.
"""

from .answer import seed_answer
from .gitea import seed_gitea

__all__ = [
    "seed_answer",
    "seed_gitea",
]
