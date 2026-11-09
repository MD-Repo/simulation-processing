"""
Author : Nick Eddy <eddy.nicholas@gmail.com>
Date   : 2026-11-09
Purpose: Publish a command's output only once it has passed its checks

Each file-writing command checks what it wrote -- a PDB's atom count, a
.psf's bonds, a cut .tpr's atoms, a converted trajectory's mapping -- and
refuses with exit 1 when the check fails. Written straight to --out, the
file under check was the published one: a refusal on a reused --out had
already replaced the caller's previous file with the one being refused, and
the cleanup in main(), which removes only what the run created, could not
put it back.

staged() hands the command a path beside the output, carrying the output's
suffix (MDAnalysis, ParmEd and gmx pick a format by it), to write and check.
Leaving the block cleanly moves the file onto the output in one rename, so
the output is whole or untouched; leaving it by an exception removes the
staged file and leaves the output as it was, absent or a previous run's.
"""

import contextlib
import os
from typing import Iterator


def staged_path(out: str) -> str:
    """Where staged() writes: the output's name with .part before its suffix,
    so s.pdb is staged at s.part.pdb."""
    root, ext = os.path.splitext(out)
    return f"{root}.part{ext}"


@contextlib.contextmanager
def staged(out: str) -> Iterator[str]:
    """Yield the path to write and check in place of out. A clean exit
    publishes it as out; an exception removes it and leaves out as it was.
    Writing nothing publishes nothing."""
    tmp = staged_path(out)
    try:
        yield tmp
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    if os.path.exists(tmp):
        os.replace(tmp, out)
