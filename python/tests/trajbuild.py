"""Builders for small trajectories the tests write from scratch: XTC frames of
a few atoms (stored raw, under the compression threshold), TRR frames in
either precision, and CHARMM DCDs. Each coordinate carries its frame
number, so a test can tell which frames a piece holds."""

import struct


def xtc(times, natoms=3, step0=0, coord=None):
    """An XTC of len(times) frames. Every coordinate in frame f is coord(f),
    or f+1."""

    out = bytearray()
    for f, t in enumerate(times):
        c = float(coord(f) if coord else f + 1)
        out += struct.pack(">iiif", 1995, natoms, step0 + f, t)
        out += struct.pack(">9f", 1, 0, 0, 0, 1, 0, 0, 0, 1)
        out += struct.pack(">i", natoms)
        out += struct.pack(">%df" % (3 * natoms), *([c] * (3 * natoms)))
    return bytes(out)


def trr(times, natoms=2, double=False, coord=None):
    """A TRR of len(times) frames with a box and coordinates, in float or
    double precision."""

    real = 8 if double else 4
    fmt = "d" if double else "f"
    out = bytearray()
    for f, t in enumerate(times):
        c = float(coord(f) if coord else f + 1)
        sizes = [0] * 10
        sizes[2] = 9 * real  # box
        sizes[7] = 3 * natoms * real  # x
        out += struct.pack(">iii", 1993, 13, 12) + b"GMX_trn_file"
        out += struct.pack(">10i", *sizes)
        out += struct.pack(">iii", natoms, f, 0)
        out += struct.pack(">" + fmt * 2, t, 0.0)  # t, lambda
        out += struct.pack(">9" + fmt, 1, 0, 0, 0, 1, 0, 0, 0, 1)
        out += struct.pack(">%d%s" % (3 * natoms, fmt), *([c] * (3 * natoms)))
    return bytes(out)


def dcd(n, frames, delta=20.4548, nsavc=1, little=True, coord=None):
    """A CHARMM DCD of n atoms whose every coordinate in frame f is coord(f),
    or f+1, with the given time step (AKMA) and save interval."""

    o = "<" if little else ">"
    body = bytearray(84)
    body[0:4] = b"CORD"
    struct.pack_into(o + "i", body, 4, frames)
    struct.pack_into(o + "i", body, 4 + 4 * 2, nsavc)
    struct.pack_into(o + "f", body, 4 + 4 * 9, delta)
    struct.pack_into(o + "i", body, 4 + 4 * 19, 24)
    out = bytearray()
    out += struct.pack(o + "i", 84) + body + struct.pack(o + "i", 84)
    out += struct.pack(o + "ii", 84, 1) + bytes(80) + struct.pack(o + "i", 84)
    out += struct.pack(o + "iii", 4, n, 4)
    for f in range(frames):
        c = float(coord(f) if coord else f + 1)
        for _ in range(3):
            out += struct.pack(o + "i", 4 * n)
            out += struct.pack(o + "%df" % n, *([c] * n))
            out += struct.pack(o + "i", 4 * n)
    return bytes(out)


def xtc_times(data):
    """The time stamps of an XTC built by xtc()."""
    import io
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import trajframes as tf

    fh = io.BytesIO(data)
    return [f.time for f in tf.walk_xtc(fh, len(data), "x.xtc")]
