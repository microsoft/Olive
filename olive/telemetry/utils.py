# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import traceback


def _format_exception_msg(ex: Exception) -> str:
    folder = "Olive"
    file_line = 'File "'
    ex = traceback.format_exception(ex, limit=5)
    lines = []
    for line in ex:
        line_trunc = line.strip()
        if line_trunc.startswith(file_line) and folder in line_trunc:
            idx = line_trunc.find(folder)
            if idx != -1:
                line_trunc = line_trunc[idx + len(folder) :]
        elif line_trunc.startswith(file_line):
            idx = line_trunc[len(file_line) :].find('"')
            line_trunc = line_trunc[idx + len(file_line) :]
        lines.append(line_trunc)
    return "\n".join(lines)
