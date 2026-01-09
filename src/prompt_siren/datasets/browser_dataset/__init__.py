# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Browser-based datasets for web agent tasks.

This module provides datasets with different observation modalities:
- browser-screenshot: Screenshot observations with coordinate-based tools
- browser-a11y: Accessibility tree observations with ARIA role-based tools
- browser-html: HTML observations with CSS selector-based tools

All datasets share the same tasks across multiple website containers
(Gitea, Apache Answer, Wiki.js, VWA Classifieds) with support for prompt
injection attacks via user-generated content, form inputs, and visual elements.
"""

from .a11y_dataset import (
    AccessibilityTreeBrowserDataset,
    create_a11y_browser_dataset,
)
from .config import BrowserDatasetConfig
from .html_dataset import (
    create_html_browser_dataset,
    HTMLBrowserDataset,
    HTMLDatasetConfig,
)
from .screenshot_dataset import (
    create_screenshot_browser_dataset,
    ScreenshotBrowserDataset,
)

__all__ = [
    # Accessibility tree-based
    "AccessibilityTreeBrowserDataset",
    # Config
    "BrowserDatasetConfig",
    # HTML-based
    "HTMLBrowserDataset",
    "HTMLDatasetConfig",
    # Screenshot-based
    "ScreenshotBrowserDataset",
    # Factory functions
    "create_a11y_browser_dataset",
    "create_html_browser_dataset",
    "create_screenshot_browser_dataset",
]
