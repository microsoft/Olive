from __future__ import annotations

# pylint: disable=relative-beyond-top-level
from .presets import gr
from .utils import convert_asis, convert_mdtext, detect_converted_mark


def postprocess(self, y: list[tuple[str | None, str | None]]) -> list[tuple[str | None, str | None]]:
    """Each message and response should be a string, which may be in Markdown format.

    Returns:
        List of tuples representing the message and response.
        Each message and response will be a string of HTML.

    """
    if y is None or y == []:
        return []
    temp = []
    for x in y:
        user, bot = x
        if not detect_converted_mark(user):
            user = convert_asis(user)
        if not detect_converted_mark(bot):
            bot = convert_mdtext(bot)
        temp.append((user, bot))
    return temp


GradioTemplateResponseOriginal = gr.routes.templates.TemplateResponse
