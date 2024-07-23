# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Based on https://github.com/lydell/json-stringify-pretty-compact/blob/main/index.js
import argparse
import concurrent.futures
import json
import logging
import os
import sys
from typing import List

from lintrunner_adapters import LintMessage, LintSeverity

LINTER_CODE = "FORMAT-JSON"


def json_dumps(obj):
    """Dump json with spaces before and after braces and brackets.

    Cannot just find and replace the braces and brackets because the string representation of the object
    may contain braces and brackets in string content.
    """
    if isinstance(obj, dict):
        items = [f'"{k}": {json_dumps(v)}' for k, v in obj.items()]
        return "{ " + ", ".join(items) + " }"
    elif isinstance(obj, list):
        items = [json_dumps(item) for item in obj]
        return "[ " + ", ".join(items) + " ]"
    else:
        return json.dumps(obj)


def format_json(passed_obj, indent: int, max_line_length: int):
    indent_str = " " * indent

    def _format_json(obj, current_indent, reserved):
        # get the string representation of the object
        # add spaces before and after braces and brackets
        string = json_dumps(obj)

        # if the string is short enough, return it
        length = max_line_length - len(current_indent) - reserved
        if len(string) <= length:
            return string

        if isinstance(obj, (dict, list)):
            # break the object into multiple lines
            next_indent = current_indent + indent_str

            items = []
            start, end = ("[", "]") if isinstance(obj, list) else ("{", "}")
            if isinstance(obj, list):
                for index, item in enumerate(obj):
                    serialized = _format_json(item, next_indent, 0 if index == len(obj) - 1 else 1) or "null"
                    items.append(serialized)
            else:
                for index, (key, value) in enumerate(obj.items()):
                    key_part = json.dumps(key) + ": "
                    serialized = _format_json(value, next_indent, len(key_part) + (0 if index == len(obj) - 1 else 1))
                    if serialized is not None:
                        items.append(key_part + serialized)
            if items:
                return f"{start}\n{next_indent}" + f",\n{next_indent}".join(items) + f"\n{current_indent}{end}"

        return string

    return _format_json(passed_obj, "", 0) + "\n"


def check_file(filename: str, indent: int, max_line_length: int) -> List[LintMessage]:
    with open(filename, "rb") as f:
        original = f.read().decode("utf-8")

    try:
        obj = json.loads(original)

        replacement = format_json(obj, indent, max_line_length)

        if original == replacement:
            return []

        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.WARNING,
                name="Format JSON",
                original=original,
                replacement=replacement,
                description="Run `lintrunner -a` to apply this patch.",
            )
        ]
    except Exception as e:
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="Format JSON",
                original=None,
                replacement=None,
                description=f"Failed to parse JSON: {e}",
            )
        ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Format files with ufmt (black + usort). Linter code: {LINTER_CODE}",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=4,
        help="number of spaces to indent",
    )
    parser.add_argument(
        "--max-line-length",
        type=int,
        default=80,
        help="maximum line length",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=(logging.NOTSET if args.verbose else logging.DEBUG if len(args.filenames) < 1000 else logging.INFO),
        stream=sys.stderr,
    )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=os.cpu_count(),
        thread_name_prefix="Thread",
    ) as executor:
        futures = {executor.submit(check_file, x, args.indent, args.max_line_length): x for x in args.filenames}
        for future in concurrent.futures.as_completed(futures):
            try:
                for lint_message in future.result():
                    lint_message.display()
            except Exception:
                logging.critical('Failed at "%s".', futures[future])
                raise


if __name__ == "__main__":
    main()
