# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Free-text telemetry redaction."""

import os
import re
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from datetime import date, datetime, time, timedelta
from uuid import UUID

MAX_TELEMETRY_STRING_LENGTH = 40_960
MAX_ERROR_MESSAGE_LENGTH = MAX_TELEMETRY_STRING_LENGTH

_SENSITIVE_COMPACT_KEYS = {
    "accountkey",
    "accesskey",
    "accesstoken",
    "apikey",
    "auth",
    "authorization",
    "authtoken",
    "clientsecret",
    "connectionstring",
    "credential",
    "credentials",
    "passwd",
    "password",
    "privatekey",
    "pwd",
    "sastoken",
    "secret",
    "secretkey",
    "servicecredential",
    "sig",
    "signature",
    "subscriptionkey",
    "token",
}
_ENVIRONMENT_COMPACT_KEYS = {"env", "environment", "environmentvariables", "environmentvars", "envvariables", "envvars"}
_PATH_KEY_SUFFIXES = ("dir", "dirs", "file", "files", "path", "paths")


def normalize_config_key_for_telemetry(key) -> str:
    if key is None:
        return ""
    value = str(key)
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def is_sensitive_config_key_for_telemetry(key) -> bool:
    normalized = normalize_config_key_for_telemetry(key)
    compact = normalized.replace("_", "")
    parts = set(normalized.split("_"))
    if compact in _SENSITIVE_COMPACT_KEYS:
        return True
    if parts & {"auth", "authorization", "credential", "credentials", "passwd", "password", "pwd", "secret", "token"}:
        return True
    if "key" in parts and parts & {"access", "account", "api", "private", "secret", "subscription"}:
        return True
    if {"connection", "string"} <= parts:
        return True
    if compact.endswith(
        ("authorization", "credential", "credentials", "passwd", "password", "pwd", "secret", "signature", "token")
    ):
        return True
    return any(
        marker in compact
        for marker in (
            "accesskey",
            "accountkey",
            "apikey",
            "connectionstring",
            "privatekey",
            "secretkey",
            "subscriptionkey",
        )
    )


def is_environment_config_key_for_telemetry(key) -> bool:
    normalized = normalize_config_key_for_telemetry(key)
    return normalized.replace("_", "") in _ENVIRONMENT_COMPACT_KEYS or bool(
        set(normalized.split("_")) & {"env", "environment"}
    )


def is_path_like_config_key_for_telemetry(key) -> bool:
    compact = normalize_config_key_for_telemetry(key).replace("_", "")
    return bool(compact) and compact.endswith(_PATH_KEY_SUFFIXES)


def scrub_config_snapshot_for_telemetry(value, key=None):
    """Recursively scrub a JSON-compatible configuration snapshot by key and value."""
    if is_sensitive_config_key_for_telemetry(key) or is_environment_config_key_for_telemetry(key):
        return "<secret>"
    if is_path_like_config_key_for_telemetry(key):
        return "<resource>"
    if isinstance(value, Mapping):
        items = {}
        collisions = set()
        for child_key, child_value in value.items():
            safe_key = scrub_string_for_telemetry(str(child_key))
            if not safe_key:
                continue
            if safe_key in items:
                collisions.add(safe_key)
            else:
                items[safe_key] = scrub_config_snapshot_for_telemetry(child_value, child_key)
        return {safe_key: items[safe_key] for safe_key in sorted(items) if safe_key not in collisions}
    if isinstance(value, (list, tuple)):
        return [scrub_config_snapshot_for_telemetry(item, key) for item in value]
    if isinstance(value, str):
        return scrub_string_for_telemetry(value)
    return value


def _token_start(value: str, index: int) -> int:
    while index > 0 and not value[index - 1].isspace() and value[index - 1] not in "\"'":
        index -= 1
    return index


def _is_drive_path_anchor(value: str, index: int) -> bool:
    char = value[index]
    if not char.isascii() or not char.isalpha():
        return False
    if index > 0 and value[index - 1] not in "\"' \t=([{,;":
        return False
    if index + 2 >= len(value):
        return False
    return value[index + 1] == ":" and value[index + 2] in "/\\"


def _is_sensitive_slash_option(value: str, index: int) -> bool:
    key_start = index + 1
    if key_start >= len(value) or not value[key_start].isascii() or not value[key_start].isalpha():
        return False
    key_end = key_start
    while key_end < len(value) and value[key_end].isascii() and (value[key_end].isalnum() or value[key_end] in "_.-"):
        key_end += 1
    if key_end == key_start or not is_sensitive_config_key_for_telemetry(value[key_start:key_end]):
        return False
    separator = key_end
    while separator < len(value) and value[separator].isspace():
        separator += 1
    return separator < len(value) and (value[separator] in "=:" or separator > key_end)


def _find_path_anchor(value: str):
    index = 0
    slash_token_end = 0
    slash_token_start = 0
    slash_token_analyzed = False
    relative_slash_anchor = None
    while index < len(value):
        char = value[index]
        if value.startswith(("./", "../", ".\\", "..\\"), index) and (
            index == 0 or value[index - 1].isspace() or value[index - 1] in "\"'=([{,;"
        ):
            return index
        if char == ":" and index + 2 < len(value) and value[index + 1 : index + 3] == "//":
            scheme_start = index
            while scheme_start > 0 and (
                value[scheme_start - 1].isascii()
                and (value[scheme_start - 1].isalnum() or value[scheme_start - 1] in "+-.")
            ):
                scheme_start -= 1
            return scheme_start
        if char == "\\" and index + 1 < len(value) and value[index + 1] == "\\":
            return index
        if char == "~" and index + 1 < len(value) and value[index + 1] in "/\\":
            return index
        if _is_drive_path_anchor(value, index):
            return index
        if char == "\\":
            if (
                index + 1 < len(value)
                and value[index + 1] not in "\\\r\n \t"
                and (index == 0 or value[index - 1] in "\"' \t=([{,;")
            ):
                return index
            separators = 0
            for candidate in value[index:]:
                if candidate in "\r\n":
                    break
                if candidate == "\\":
                    separators += 1
                    if separators >= 2:
                        return _token_start(value, index)
        if char == "/":
            if index >= slash_token_end:
                slash_token_start = _token_start(value, index)
                slash_token_end = index
                while (
                    slash_token_end < len(value)
                    and not value[slash_token_end].isspace()
                    and value[slash_token_end] not in "\"'"
                ):
                    slash_token_end += 1
                slash_token_analyzed = False
            if (
                index + 1 < len(value)
                and value[index + 1] not in "/\r\n \t"
                and (index == 0 or value[index - 1] in "\"' \t=([{,;")
                and not _is_sensitive_slash_option(value, index)
            ):
                return index
            if not slash_token_analyzed:
                slash_token_analyzed = True
                segments = 0
                cursor = index
                while cursor < slash_token_end and value[cursor] == "/":
                    separator_end = cursor + 1
                    while separator_end < slash_token_end and value[separator_end] == "/":
                        separator_end += 1
                    cursor = separator_end
                    segment_start = cursor
                    while cursor < slash_token_end and value[cursor] not in "/\r\n \t":
                        cursor += 1
                    if cursor == segment_start:
                        break
                    segments += 1
                if segments >= 2:
                    return relative_slash_anchor if relative_slash_anchor is not None else slash_token_start
                if segments == 1 and slash_token_start < index:
                    if relative_slash_anchor is not None:
                        return relative_slash_anchor
                    token = value[slash_token_start:slash_token_end]
                    if token.lower() not in {"and/or", "n/a", "read/write"} and any(char.isalpha() for char in token):
                        relative_slash_anchor = slash_token_start
        index += 1
    return None


def _is_secret_key_char(char: str) -> bool:
    return char.isascii() and (char.isalnum() or char in "_.-")


def _is_secret_key_boundary(char: str) -> bool:
    return char.isspace() or char in "?&#;,\"'([{/-"


def _find_sensitive_value_anchor(value: str):
    index = 0
    while index < len(value):
        char = value[index]
        if not char.isascii() or not char.isalpha():
            index += 1
            continue
        if index > 0 and not _is_secret_key_boundary(value[index - 1]):
            index += 1
            continue

        key_end = index + 1
        while key_end < len(value) and _is_secret_key_char(value[key_end]):
            key_end += 1
        if not is_sensitive_config_key_for_telemetry(value[index:key_end]):
            index = key_end
            continue

        separator = key_end
        if separator < len(value) and value[separator] in "\"'":
            separator += 1
        before_whitespace = separator
        while separator < len(value) and value[separator].isspace():
            separator += 1
        assignment = separator < len(value) and value[separator] in "=:"
        cli_option = index > 0 and value[index - 1] in "-/"
        delimited_cli_value = False
        if cli_option and not assignment:
            separator = key_end
            while separator < len(value) and (value[separator].isspace() or value[separator] in "\"',[](){}"):
                if value[separator] in "\"',[](){}":
                    delimited_cli_value = True
                separator += 1
        separated_cli_value = cli_option and before_whitespace < separator < len(value)
        separated_cli_value = separated_cli_value and (value[separator] != "-" or delimited_cli_value)
        if not assignment and not separated_cli_value:
            index = key_end
            continue

        value_start = separator + 1 if assignment else separator
        while value_start < len(value) and value[value_start].isspace():
            value_start += 1
        if value_start < len(value) and value[value_start] not in "&;\r\n":
            return value_start
        index = key_end
    return None


def _is_user_info_terminator(char: str) -> bool:
    return char.isspace() or char in '"\\/?#[]{}'


def _is_authority_terminator(char: str) -> bool:
    return char.isspace() or char in "\"')},;/?#"


def _find_credential_url_anchor(value: str):
    token_start = 0
    colon = None
    index = 0
    while index < len(value):
        char = value[index]
        if colon is None:
            if _is_user_info_terminator(char):
                token_start = index + 1
            elif char == ":" and index > token_start:
                colon = index
            index += 1
            continue

        if _is_user_info_terminator(char):
            token_start = index + 1
            colon = None
            index += 1
            continue
        if char != "@" or colon + 1 == index:
            index += 1
            continue

        host_start = index + 1
        if host_start == len(value):
            index += 1
            continue
        if value[host_start] == "[":
            host_end = value.find("]", host_start + 1)
            if host_end > host_start + 1:
                return token_start
            index += 1
            continue

        host_end = host_start
        while host_end < len(value) and not _is_authority_terminator(value[host_end]):
            host_end += 1
        if host_end > host_start:
            return token_start
        index = host_end + 1
        token_start = index
        colon = None
    return None


def _truncate_utf8(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _scrub_string_for_telemetry(value: str, max_bytes: int) -> str:
    path_anchor = _find_path_anchor(value)
    secret_value = _find_sensitive_value_anchor(value)
    credential_anchor = _find_credential_url_anchor(value)
    anchors = (
        (path_anchor, "[path]"),
        (secret_value, "[secret]"),
        (credential_anchor, "[secret]"),
    )
    anchor, marker = min(
        ((anchor, marker) for anchor, marker in anchors if anchor is not None),
        default=(None, None),
    )
    if anchor is not None:
        return _truncate_utf8(value[:anchor], max_bytes - len(marker)) + marker
    return _truncate_utf8(value, max_bytes)


def scrub_string_for_telemetry(value: str) -> str:
    """Redact and cap a general telemetry string."""
    return _scrub_string_for_telemetry(value, MAX_TELEMETRY_STRING_LENGTH)


def scrub_error_message_for_telemetry(value: str) -> str:
    """Redact and cap an error message at 40,960 UTF-8 bytes."""
    return _scrub_string_for_telemetry(value, MAX_ERROR_MESSAGE_LENGTH)


def scrub_value_for_telemetry(value, key=None):
    """Recursively scrub strings and path-like values before serialization."""
    if is_sensitive_config_key_for_telemetry(key) or is_environment_config_key_for_telemetry(key):
        return "<secret>"
    if is_path_like_config_key_for_telemetry(key):
        return "<resource>"
    if isinstance(value, os.PathLike):
        return "[path]"
    if isinstance(value, str):
        return scrub_string_for_telemetry(value)
    if isinstance(value, (bytes, bytearray)):
        try:
            return scrub_string_for_telemetry(bytes(value).decode("utf-8"))
        except UnicodeDecodeError:
            return "[binary]"
    if value is None or isinstance(
        value,
        (bool, int, float, datetime, date, time, timedelta, UUID),
    ):
        return value
    if isinstance(value, Mapping):
        items = {}
        collisions = set()
        for child_key, child in value.items():
            if isinstance(child_key, os.PathLike):
                safe_key = "[path]"
            elif isinstance(child_key, str):
                safe_key = scrub_string_for_telemetry(child_key)
            else:
                try:
                    safe_key = scrub_string_for_telemetry(str(child_key))
                except Exception:
                    safe_key = f"[unsupported:{type(child_key).__name__}]"
            if safe_key:
                if safe_key in items:
                    collisions.add(safe_key)
                else:
                    items[safe_key] = scrub_value_for_telemetry(child, child_key)
        return {safe_key: items[safe_key] for safe_key in sorted(items) if safe_key not in collisions}
    if isinstance(value, list):
        return [scrub_value_for_telemetry(child, key) for child in value]
    if isinstance(value, tuple):
        return tuple(scrub_value_for_telemetry(child, key) for child in value)
    if isinstance(value, AbstractSet):
        children = [scrub_value_for_telemetry(child, key) for child in value]
        return sorted(children, key=lambda child: (type(child).__name__, repr(child)))
    try:
        return scrub_string_for_telemetry(str(value))
    except Exception:
        return f"[unsupported:{type(value).__name__}]"
