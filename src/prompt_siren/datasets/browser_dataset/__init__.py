# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Browser-based dataset for web agent tasks.

This dataset provides tasks across multiple website containers (Gitea, Apache Answer,
Wiki.js, VWA Classifieds) with support for prompt injection attacks via user-generated
content, form inputs, and visual elements.
"""

from .config import BrowserDatasetConfig
from .dataset import BrowserDataset, create_browser_dataset

__all__ = [
    "BrowserDataset",
    "BrowserDatasetConfig",
    "create_browser_dataset",
]
