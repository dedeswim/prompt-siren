# Copyright (c) Meta Platforms, Inc. and affiliates.
"""CSS selector-based browser tools for HTML observations.

These tools are designed for agents that receive HTML content and
interact using CSS selectors.
"""

from pydantic_ai import RunContext

from ....environments.browser_env import BrowserEnvState


async def click_selector(
    ctx: RunContext[BrowserEnvState],
    selector: str,
) -> str:
    """Click on an element matching a CSS selector.

    Args:
        ctx: The run context containing the browser page
        selector: CSS selector to find the element

    Returns:
        Status message describing the click action
    """
    page = ctx.deps.page
    try:
        await page.click(selector, timeout=5000)
        return f"Clicked element matching selector: {selector}"
    except Exception as e:
        return f"Failed to click selector '{selector}': {e}"


async def fill_input(
    ctx: RunContext[BrowserEnvState],
    selector: str,
    value: str,
) -> str:
    """Fill an input field matching a CSS selector.

    Args:
        ctx: The run context containing the browser page
        selector: CSS selector to find the input element
        value: Value to fill in the input

    Returns:
        Status message describing the fill action
    """
    page = ctx.deps.page
    try:
        await page.fill(selector, value, timeout=5000)
        return f"Filled input '{selector}' with: {value[:50]}{'...' if len(value) > 50 else ''}"
    except Exception as e:
        return f"Failed to fill input '{selector}': {e}"


async def get_page_text(
    ctx: RunContext[BrowserEnvState],
) -> str:
    """Get all visible text content from the page.

    Args:
        ctx: The run context containing the browser page

    Returns:
        The text content of the page
    """
    page = ctx.deps.page
    text = await page.inner_text("body")
    # Truncate if too long
    max_length = 10000
    if len(text) > max_length:
        text = text[:max_length] + "\n...[truncated]"
    return text


async def scroll_to_element(
    ctx: RunContext[BrowserEnvState],
    selector: str,
) -> str:
    """Scroll to bring an element into view.

    Args:
        ctx: The run context
        selector: CSS selector for the element

    Returns:
        Status message
    """
    page = ctx.deps.page
    try:
        await page.locator(selector).scroll_into_view_if_needed(timeout=5000)
        return f"Scrolled to '{selector}'"
    except Exception as e:
        return f"Failed to scroll to '{selector}': {e}"
