# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Browser interaction tools for the browser dataset.

These tools allow agents to interact with web pages via Playwright.
"""

from typing import Literal

from pydantic_ai import RunContext

from ...environments.browser_env import BrowserEnvState


async def click(
    ctx: RunContext[BrowserEnvState],
    x: int,
    y: int,
    button: Literal["left", "right", "middle"] = "left",
) -> str:
    """Click at a specific position on the page.

    Args:
        ctx: The run context containing the browser page
        x: X coordinate to click
        y: Y coordinate to click
        button: Mouse button to use (left, right, or middle)

    Returns:
        Status message describing the click action
    """
    page = ctx.deps.page
    await page.mouse.click(x, y, button=button)
    return f"Clicked at ({x}, {y}) with {button} button"


async def scroll(
    ctx: RunContext[BrowserEnvState],
    x: int,
    y: int,
    scroll_x: int,
    scroll_y: int,
) -> str:
    """Scroll the page by a specified amount.

    Args:
        ctx: The run context containing the browser page
        x: X coordinate to scroll from
        y: Y coordinate to scroll from
        scroll_x: Horizontal scroll amount (positive = right)
        scroll_y: Vertical scroll amount (positive = down)

    Returns:
        Status message describing the scroll action
    """
    page = ctx.deps.page
    await page.mouse.move(x, y)
    await page.evaluate(f"window.scrollBy({scroll_x}, {scroll_y})")
    return f"Scrolled by ({scroll_x}, {scroll_y}) from position ({x}, {y})"


async def type_text(
    ctx: RunContext[BrowserEnvState],
    text: str,
) -> str:
    """Type text using the keyboard.

    Args:
        ctx: The run context containing the browser page
        text: Text to type

    Returns:
        Status message describing the typing action
    """
    page = ctx.deps.page
    await page.keyboard.type(text)
    return f"Typed: {text[:50]}{'...' if len(text) > 50 else ''}"


async def press_key(
    ctx: RunContext[BrowserEnvState],
    key: str,
) -> str:
    """Press a keyboard key.

    Args:
        ctx: The run context containing the browser page
        key: Key to press (e.g., "Enter", "Tab", "Escape", "ArrowDown")

    Returns:
        Status message describing the key press
    """
    page = ctx.deps.page
    await page.keyboard.press(key)
    return f"Pressed key: {key}"


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


async def goto_url(
    ctx: RunContext[BrowserEnvState],
    url: str,
) -> str:
    """Navigate to a specific URL.

    Args:
        ctx: The run context containing the browser page
        url: URL to navigate to

    Returns:
        Status message with the final URL
    """
    page = ctx.deps.page
    await page.goto(url, timeout=30000)
    return f"Navigated to: {page.url}"


async def go_back(
    ctx: RunContext[BrowserEnvState],
) -> str:
    """Go back to the previous page in browser history.

    Args:
        ctx: The run context containing the browser page

    Returns:
        Status message with the current URL
    """
    page = ctx.deps.page
    await page.go_back()
    return f"Went back. Current URL: {page.url}"


async def go_forward(
    ctx: RunContext[BrowserEnvState],
) -> str:
    """Go forward to the next page in browser history.

    Args:
        ctx: The run context containing the browser page

    Returns:
        Status message with the current URL
    """
    page = ctx.deps.page
    await page.go_forward()
    return f"Went forward. Current URL: {page.url}"
