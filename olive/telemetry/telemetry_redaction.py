# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Free-text telemetry redaction."""

MAX_TELEMETRY_STRING_LENGTH = 256
MAX_ERROR_MESSAGE_LENGTH = 40_960


def _token_start(value: str, index: int) -> int:
    while index > 0 and not value[index - 1].isspace() and value[index - 1] not in "\"'":
        index -= 1
    return index


def _find_path_anchor(value: str):
    for index, char in enumerate(value):
        if char == "\\" and index + 1 < len(value) and value[index + 1] == "\\":
            return index
        if char == "~" and index + 1 < len(value) and value[index + 1] in "/\\":
            return index
        if (
            char.isascii()
            and char.isalpha()
            and index + 2 < len(value)
            and value[index + 1] == ":"
            and value[index + 2] in "/\\"
        ):
            return index
        if char == "\\":
            separators = 0
            for candidate in value[index:]:
                if candidate in "\r\n":
                    break
                if candidate == "\\":
                    separators += 1
                    if separators >= 2:
                        return _token_start(value, index)
        if char == "/":
            segments = 0
            cursor = index
            while cursor < len(value) and value[cursor] == "/":
                while cursor < len(value) and value[cursor] == "/":
                    cursor += 1
                segment_start = cursor
                while cursor < len(value) and value[cursor] not in "/\r\n \t":
                    cursor += 1
                if cursor == segment_start:
                    break
                segments += 1
            if segments >= 2:
                return _token_start(value, index)
    return None


def _truncate_utf8(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _scrub_string_for_telemetry(value: str, max_bytes: int) -> str:
    anchor = _find_path_anchor(value)
    scrubbed = value if anchor is None else value[:anchor] + "[path]"
    return _truncate_utf8(scrubbed, max_bytes)


def scrub_string_for_telemetry(value: str) -> str:
    """Redact and cap a general telemetry string."""
    return _scrub_string_for_telemetry(value, MAX_TELEMETRY_STRING_LENGTH)


def scrub_error_message_for_telemetry(value: str) -> str:
    """Redact and cap an error message at 40,960 UTF-8 bytes."""
    return _scrub_string_for_telemetry(value, MAX_ERROR_MESSAGE_LENGTH)
