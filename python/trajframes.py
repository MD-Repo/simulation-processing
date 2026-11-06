"""
Author : Nick Eddy <eddy.nicholas@gmail.com>
Date   : 2026-11-06
Purpose: Walk a trajectory's frames without decoding its coordinates, for the
         commands that rewrite its time axis or regroup its frames

Frame headers are walked directly rather than through a trajectory library:
what the commands need is where each frame begins and ends and where its
time stamp sits, so that a frame can be copied byte for byte or its stamp
rewritten in place, with every coordinate left as the release wrote it. The
walk skips each coordinate block by its stored length, so no coordinates
are ever decompressed and the cost is a seek per frame.

  * XTC: a 16-byte header per frame (magic 1995, atoms, step, time), a box,
    then a coordinate block whose length is stored; frames of nine atoms or
    fewer are stored as raw floats. Big-endian throughout.
  * TRR: a fixed header stating the size of every block that follows, from
    which the precision (float or double) is told; the time is the first
    real after the header. Big-endian.
  * DCD: one header, then frames of fixed size; the header states the time
    step and the save interval, and no frame carries a time. Byte order from
    the first record marker.
  * AMBER NetCDF (classic, 64-bit offset or 64-bit data): the "time"
    record variable's values sit one per record at a fixed stride.
  * HDF5 (mdtraj, H5MD, NetCDF-4): the time dataset is read and written
    through h5py; a dataset behind a compression filter is not rewritten.
  * DTR: a directory, whose frames store their time twice; refused.

Ported from mdrepo-agent's internal/traj (xtc.go, trr.go, dcd.go, netcdf.go,
splice.go) when the trajectory rewrites moved here (mdrepo-agent#98). The
agent keeps its own walkers for measuring; these feed rescale_times.py,
set_dcd_spacing.py, cut_replicates.py, join_trajectories.py and
split_at_resets.py.
"""

import hashlib
import math
import os
import struct
from dataclasses import dataclass, field
from typing import BinaryIO, Optional

XTC_MAGIC = 1995
TRR_MAGIC = 1993
TRR_VERSION = b"GMX_trn_file"
XTC_HEADER = 16
XTC_TIME_OFFSET = 12
XTC_BOX = 36
XTC_COMPRESSED_PREAMBLE = 4 + 4 * 6 + 4  # precision, minint[3], maxint[3], smallidx
XTC_RAW_ATOM_LIMIT = 9
TRR_FIXED_HEADER = 4 + 4 + 4 + 12 + 10 * 4 + 4 + 4 + 4
DCD_HEADER_RECORD = 84
DCD_CELL_RECORD = 48
DCD_DELTA_OFFSET = 4 + 4 + 4 * 9
DCD_NSET_OFFSET = 4 + 4
AKMA_PS = 0.04888821

# Two stamps are the same instant within a thousandth of a picosecond, or a
# millionth of the time itself, well above what a float32 stamp rounds away.


def same_time(a: float, b: float) -> bool:
    return abs(a - b) <= max(1e-3, abs(b) * 1e-6)


class BadTrajectory(ValueError):
    """A file that is not a trajectory this can walk, or is damaged."""


class Refused(ValueError):
    """An operation the format does not allow: a fact about the file, not a
    failure. The caller discloses it instead of retrying."""


# --------------------------------------------------
def identify(path: str) -> Optional[str]:
    """The format of the file at path by its first bytes: xtc, trr, dcd,
    netcdf, hdf5, dtr, or None."""

    if os.path.isdir(path):
        return "dtr" if os.path.exists(os.path.join(path, "clickme.dtr")) else None
    with open(path, "rb") as fh:
        head = fh.read(8)
    if len(head) >= 4:
        magic = struct.unpack(">i", head[:4])[0]
        if magic == XTC_MAGIC:
            return "xtc"
        if magic == TRR_MAGIC:
            return "trr"
        if head[:4] == b"CDF\x01" or head[:4] == b"CDF\x02" or head[:4] == b"CDF\x05":
            return "netcdf"
        if head == b"\x89HDF\r\n\x1a\n":
            return "hdf5"
        if (
            len(head) == 8
            and head[4:8] == b"CORD"
            and DCD_HEADER_RECORD
            in (struct.unpack("<i", head[:4])[0], struct.unpack(">i", head[:4])[0])
        ):
            return "dcd"
    return None


# --------------------------------------------------
@dataclass
class Frame:
    """One frame: its byte range, and where its time stamp sits."""

    start: int
    end: int
    time: Optional[float] = None
    time_offset: Optional[int] = None
    wide: bool = False  # float64 rather than float32
    order: str = ">"


@dataclass
class DCDHeader:
    order: str
    natoms: int
    nset: int
    istart: int
    nsavc: int
    delta: float
    charmm: bool
    crystal: bool
    four_dim: bool
    header_bytes: int
    frame_bytes: int

    def declared_ps_per_frame(self) -> float:
        return max(self.nsavc, 1) * self.delta * AKMA_PS

    def placeholder(self) -> bool:
        """A header whose time fields were never filled in: a time step of
        exactly 1.0 in either unit, or a step and interval that multiply to
        exactly 1 ps."""
        return (
            self.delta == 1
            or is_default_spacing(self.delta * AKMA_PS)
            or is_default_spacing(self.declared_ps_per_frame())
        )


def is_default_spacing(ps: float) -> bool:
    """Exactly 1 ps, as float32 rounds it: the value every writer falls back
    to when nobody tells it the spacing."""
    return abs(ps - 1.0) <= 1e-6


@dataclass
class Run:
    """A trajectory as a run of frames."""

    format: str
    name: str
    frames: list[Frame]
    size: int
    head: bytes = b""  # a DCD's header
    dcd: Optional[DCDHeader] = None
    # How many bytes at the head of each frame say where it sits in the run
    # (an XTC's step and time, a TRR's header) rather than what it holds.
    skip: int = 0

    @property
    def times(self) -> Optional[list[float]]:
        if self.format not in ("xtc", "trr"):
            return None
        return [f.time for f in self.frames]  # type: ignore[misc]

    def byte_range(self, start: int, stop: int) -> tuple[int, int]:
        """The byte range of frames [start, stop)."""
        end = self.frames[stop].start if stop < len(self.frames) else self.size
        return self.frames[start].start, end

    def digest(self, fh: BinaryIO, i: int) -> str:
        """A frame's coordinates, and nothing its header says about where it
        sits in the run: two files holding the same frame digest alike."""
        a, b = self.byte_range(i, i + 1)
        a += self.skip
        if a > b:
            raise BadTrajectory(f"frame {i} of {self.name} is shorter than its header")
        fh.seek(a)
        return hashlib.sha1(fh.read(b - a)).hexdigest()


# --------------------------------------------------
def _read_exact(fh: BinaryIO, n: int) -> bytes:
    b = fh.read(n)
    if len(b) != n:
        raise EOFError
    return b


def walk_xtc(fh: BinaryIO, size: int, name: str) -> list[Frame]:
    frames: list[Frame] = []
    natoms0 = None
    while True:
        offset = fh.tell()
        head = fh.read(XTC_HEADER)
        if len(head) < XTC_HEADER:
            break  # the normal end of the file
        magic, natoms, step, t = struct.unpack(">iiif", head)
        if magic != XTC_MAGIC:
            raise BadTrajectory(
                f"{name}: bad frame magic {magic} at byte {offset} (frame {len(frames)})"
            )
        if natoms0 is None:
            natoms0 = natoms
        elif natoms != natoms0:
            raise BadTrajectory(
                f"{name}: atom count changes mid-file ({natoms0} -> {natoms} at frame {len(frames)})"
            )
        try:
            fh.seek(XTC_BOX, os.SEEK_CUR)
            lsize = struct.unpack(">i", _read_exact(fh, 4))[0]
            if lsize != natoms:
                raise BadTrajectory(
                    f"{name}: coordinate block declares {lsize} atoms, header says {natoms} (frame {len(frames)})"
                )
            if lsize <= XTC_RAW_ATOM_LIMIT:
                skip = lsize * 3 * 4
            else:
                fh.seek(XTC_COMPRESSED_PREAMBLE, os.SEEK_CUR)
                nbytes = struct.unpack(">i", _read_exact(fh, 4))[0]
                if nbytes < 0:
                    raise BadTrajectory(
                        f"{name}: negative block length at frame {len(frames)}"
                    )
                skip = nbytes + (4 - nbytes % 4) % 4
        except EOFError:
            raise BadTrajectory(f"{name}: truncated at frame {len(frames)}")
        pos = fh.seek(skip, os.SEEK_CUR)
        if pos > size:
            raise BadTrajectory(f"{name}: truncated at frame {len(frames)}")
        frames.append(Frame(offset, pos, t, offset + XTC_TIME_OFFSET, False, ">"))
    if not frames:
        raise BadTrajectory(f"{name}: no readable frames")
    return frames


def walk_trr(fh: BinaryIO, size: int, name: str) -> list[Frame]:
    frames: list[Frame] = []
    natoms0 = None
    while True:
        offset = fh.tell()
        head = fh.read(TRR_FIXED_HEADER)
        if len(head) < TRR_FIXED_HEADER:
            break
        magic = struct.unpack(">i", head[:4])[0]
        if magic != TRR_MAGIC:
            raise BadTrajectory(
                f"{name}: bad frame magic {magic} at byte {offset} (frame {len(frames)})"
            )
        if head[12 : 12 + len(TRR_VERSION)] != TRR_VERSION:
            raise BadTrajectory(
                f"{name}: frame {len(frames)} names version {head[12:24]!r}, not {TRR_VERSION!r}"
            )
        sizes = struct.unpack(">10i", head[24:64])
        natoms, step, nre = struct.unpack(">iii", head[64:76])
        if any(s < 0 for s in sizes):
            raise BadTrajectory(f"{name}: negative block size at frame {len(frames)}")
        box, x, v, f = sizes[2], sizes[7], sizes[8], sizes[9]
        if box:
            real = box // 9
        elif x and natoms > 0:
            real = x // (natoms * 3)
        elif v and natoms > 0:
            real = v // (natoms * 3)
        elif f and natoms > 0:
            real = f // (natoms * 3)
        else:
            real = 0
        if real not in (4, 8):
            raise BadTrajectory(
                f"{name}: frame {len(frames)} holds no box, coordinates, velocities or forces, "
                "so its precision cannot be told"
            )
        raw = fh.read(real)
        if len(raw) != real:
            raise BadTrajectory(f"{name}: truncated at frame {len(frames)}")
        t = struct.unpack(">d" if real == 8 else ">f", raw)[0]
        pos = fh.seek(real + sum(sizes), os.SEEK_CUR)
        if pos > size:
            raise BadTrajectory(f"{name}: truncated at frame {len(frames)}")
        if natoms0 is None:
            natoms0 = natoms
        elif natoms != natoms0:
            raise BadTrajectory(
                f"{name}: atom count changes mid-file ({natoms0} -> {natoms} at frame {len(frames)})"
            )
        frames.append(Frame(offset, pos, t, offset + TRR_FIXED_HEADER, real == 8, ">"))
    if not frames:
        raise BadTrajectory(f"{name}: no readable frames")
    if natoms0 is not None and natoms0 <= 0:
        raise BadTrajectory(f"{name}: header states {natoms0} atoms")
    return frames


def read_dcd_header(fh: BinaryIO, name: str) -> DCDHeader:
    fh.seek(0)
    marker = fh.read(4)
    if len(marker) < 4:
        raise BadTrajectory(f"{name}: too short to be a DCD")
    if struct.unpack("<i", marker)[0] == DCD_HEADER_RECORD:
        order = "<"
    elif struct.unpack(">i", marker)[0] == DCD_HEADER_RECORD:
        order = ">"
    else:
        raise BadTrajectory(
            f"{name}: first record is {struct.unpack('<i', marker)[0]} bytes, not the {DCD_HEADER_RECORD} a DCD header must be"
        )
    body = fh.read(DCD_HEADER_RECORD)
    if len(body) < DCD_HEADER_RECORD:
        raise BadTrajectory(f"{name}: header record is short")
    if body[:4] != b"CORD":
        raise BadTrajectory(f'{name}: header says {body[:4]!r}, not "CORD"')
    ic = struct.unpack(order + "20i", body[4:])
    charmm = ic[19] != 0
    if charmm:
        delta = struct.unpack(order + "f", body[4 + 4 * 9 : 4 + 4 * 10])[0]
    else:
        delta = struct.unpack(order + "d", body[4 + 4 * 9 : 4 + 4 * 11])[0]

    def marker_of(what: str) -> int:
        b = fh.read(4)
        if len(b) < 4:
            raise BadTrajectory(f"{name}: ended inside a record marker")
        return struct.unpack(order + "i", b)[0]

    def expect(want: int, what: str) -> None:
        got = marker_of(what)
        if got != want:
            raise BadTrajectory(
                f"{name}: {what} record marker is {got}, expected {want}"
            )

    expect(DCD_HEADER_RECORD, "header")
    title_len = marker_of("title")
    if title_len < 4 or title_len > 4 + 80 * 1000:
        raise BadTrajectory(
            f"{name}: title record is {title_len} bytes, which is not credible"
        )
    fh.seek(title_len, os.SEEK_CUR)
    expect(title_len, "title")
    expect(4, "atom-count")
    natom_b = fh.read(4)
    if len(natom_b) < 4:
        raise BadTrajectory(f"{name}: header ends before the atom count")
    natoms = struct.unpack(order + "i", natom_b)[0]
    expect(4, "atom-count")
    header_bytes = fh.tell()
    coords = 4 + 4 * natoms + 4
    frame_bytes = 3 * coords
    crystal, four_dim = ic[10] != 0, ic[11] != 0
    if crystal:
        frame_bytes += 4 + DCD_CELL_RECORD + 4
    if four_dim:
        frame_bytes += coords
    return DCDHeader(
        order,
        natoms,
        ic[0],
        ic[1],
        ic[2],
        delta,
        charmm,
        crystal,
        four_dim,
        header_bytes,
        frame_bytes,
    )


def walk_dcd(fh: BinaryIO, size: int, name: str) -> tuple[DCDHeader, list[Frame]]:
    h = read_dcd_header(fh, name)
    if h.natoms <= 0:
        raise BadTrajectory(f"{name}: header states {h.natoms} atoms")
    body = size - h.header_bytes
    if body < 0:
        raise BadTrajectory(f"{name}: file ends inside its header")
    n, rest = divmod(body, h.frame_bytes)
    if rest:
        raise BadTrajectory(
            f"{name}: truncated: {n} complete frames then {rest} bytes of an incomplete one"
        )
    if n == 0:
        raise BadTrajectory(f"{name}: no frames")
    for i in (0, n - 1) if n > 1 else (0,):
        fh.seek(h.header_bytes + i * h.frame_bytes)
        want, what = (
            (DCD_CELL_RECORD, "unit-cell")
            if h.crystal
            else (4 * h.natoms, "coordinate")
        )
        got = struct.unpack(h.order + "i", _read_exact(fh, 4))[0]
        if got != want:
            raise BadTrajectory(
                f"{name}: frame {i + 1} begins with a {got}-byte {what} record, expected {want}; "
                "the file may use fixed atoms, which this cannot walk"
            )
    frames = [
        Frame(
            h.header_bytes + i * h.frame_bytes, h.header_bytes + (i + 1) * h.frame_bytes
        )
        for i in range(n)
    ]
    return h, frames


def read_run(path: str) -> Run:
    """Walk a DCD, XTC or TRR into its frames. Other formats cannot be cut or
    joined by whole frames."""

    name = os.path.basename(path)
    fmt = identify(path)
    if fmt not in ("dcd", "xtc", "trr"):
        raise Refused(
            f"{name} is {fmt or 'not a trajectory this reads'}, which cannot be cut or joined by whole "
            "frames; only DCD, XTC and TRR can"
        )
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        if fmt == "dcd":
            h, frames = walk_dcd(fh, size, name)
            fh.seek(0)
            return Run("dcd", name, frames, size, head=fh.read(h.header_bytes), dcd=h)
        if fmt == "xtc":
            return Run("xtc", name, walk_xtc(fh, size, name), size, skip=XTC_HEADER)
        frames = walk_trr(fh, size, name)
        real = 8 if frames[0].wide else 4
        return Run(
            "trr", name, frames, size, skip=TRR_FIXED_HEADER + 2 * real
        )  # the time and lambda


# --------------------------------------------------
# NetCDF classic: the "time" record variable's offsets.

CDF_DIMENSION, CDF_VARIABLE, CDF_ATTRIBUTE = 0x0A, 0x0B, 0x0C


class _CDF:
    def __init__(self, buf: bytes, name: str):
        self.buf, self.pos, self.name = buf, 0, name
        self.version = buf[3]
        if self.version not in (1, 2, 5):
            raise BadTrajectory(
                f"{name}: CDF version {self.version} is not classic, 64-bit offset or 64-bit data"
            )
        self.pos = 4

    def _take(self, n: int) -> bytes:
        if self.pos + n > len(self.buf):
            raise BadTrajectory(f"{self.name}: header ends early at byte {self.pos}")
        b = self.buf[self.pos : self.pos + n]
        self.pos += n
        return b

    def int32(self) -> int:
        return struct.unpack(">i", self._take(4))[0]

    def int64(self) -> int:
        return struct.unpack(">q", self._take(8))[0]

    def non_neg(self) -> int:
        return self.int64() if self.version == 5 else self.int32()

    def offset(self) -> int:
        return self.int32() if self.version == 1 else self.int64()

    def string(self) -> bytes:
        n = self.non_neg()
        s = self._take(n)
        self._take((4 - n % 4) % 4)
        return s

    def numrecs(self) -> int:
        n = self.non_neg()
        return 0 if n == 0xFFFFFFFF else n


_CDF_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 4, 6: 8, 7: 1, 8: 2, 9: 4, 10: 8, 11: 8}


def netcdf_time_stamps(fh: BinaryIO, size: int, name: str) -> list[Frame]:
    """Each frame's time stamp in an AMBER NetCDF: offset, value, width."""

    fh.seek(0)
    buf = fh.read(min(size, 1 << 24))
    r = _CDF(buf, name)
    numrecs = r.numrecs()
    # dimensions
    tag = r.int32()
    dims: list[tuple[bytes, int]] = []
    if tag == CDF_DIMENSION:
        for _ in range(r.non_neg()):
            dims.append((r.string(), r.non_neg()))
    elif tag == 0:
        r.non_neg()
    else:
        raise BadTrajectory(f"{name}: expected the dimension list")

    def attrs() -> dict[bytes, bytes]:
        tag = r.int32()
        n = r.non_neg()
        out: dict[bytes, bytes] = {}
        if tag != CDF_ATTRIBUTE:
            return out
        for _ in range(n):
            key = r.string()
            typ = r.int32()
            cnt = r.non_neg()
            raw = r._take(cnt * _CDF_SIZES.get(typ, 1))
            r._take((4 - (cnt * _CDF_SIZES.get(typ, 1)) % 4) % 4)
            out[key] = raw
        return out

    attrs()  # global attributes
    tag = r.int32()
    nvars = r.non_neg()
    if tag != CDF_VARIABLE and nvars:
        raise BadTrajectory(f"{name}: expected the variable list")
    time = None
    recsize = 0
    nrecvars = 0
    for _ in range(nvars):
        vname = r.string()
        ndims = r.non_neg()
        vdims = [r.non_neg() for _ in range(ndims)]
        vattrs = attrs()
        typ = r.int32()
        vsize = r.non_neg()
        begin = r.offset()
        is_record = bool(vdims) and dims[vdims[0]][1] == 0
        if is_record:
            nrecvars += 1
            recsize += vsize
        if vname == b"time" and is_record and len(vdims) == 1:
            time = (typ, begin, vattrs.get(b"units", b""))
    if time is None:
        raise Refused("the file carries no per-frame time variable")
    typ, begin, units = time
    if typ not in (5, 6):
        raise BadTrajectory(
            f"{name}: the time variable has type {typ}, not float or double"
        )
    # A lone record variable is packed without padding.
    if nrecvars == 1:
        recsize = _CDF_SIZES[typ]
    scale, ps = time_scale(units.decode("latin-1"))
    if not ps:
        raise Refused(
            f"the time variable is in {units.decode('latin-1')!r}, not picoseconds"
        )
    wide = typ == 6
    width = 8 if wide else 4
    frames = []
    for i in range(numrecs):
        off = begin + i * recsize
        fh.seek(off)
        raw = fh.read(width)
        if len(raw) < width:
            raise BadTrajectory(f"{name}: truncated inside frame {i}")
        t = struct.unpack(">d" if wide else ">f", raw)[0]
        frames.append(Frame(off, off + width, t, off, wide, ">"))
    return frames


def time_scale(units: str) -> tuple[float, bool]:
    """The factor from the declared unit to picoseconds, and whether the
    unit is picoseconds (the convention; anything else is read but its
    stamps are not rewritten, since the offsets hold the file's own unit)."""

    u = units.strip().lower()
    if u == "" or u.startswith("picosecond") or u == "ps":
        return 1.0, True
    if u.startswith("nanosecond") or u == "ns":
        return 1000.0, False
    if u.startswith("femtosecond") or u == "fs":
        return 0.001, False
    return 1.0, False


# --------------------------------------------------
# HDF5: the time dataset, through h5py.


def hdf5_time_dataset(path: str):
    """Open the file read-write and return (file, time dataset) for the
    mdtraj/AMBER layout (/coordinates and /time) or H5MD
    (/particles/<name>/position/{value,time}). Refuses a dataset that cannot
    be rewritten in place."""

    import h5py

    f = h5py.File(path, "r+")
    try:
        if "coordinates" in f:
            if "time" not in f:
                raise Refused("the file carries no per-frame time dataset")
            td = f["time"]
        elif "particles" in f:
            td = None
            for name, grp in f["particles"].items():
                if (
                    isinstance(grp, h5py.Group)
                    and "position" in grp
                    and "value" in grp["position"]
                ):
                    if "time" in grp["position"]:
                        td = grp["position"]["time"]
                    break
            if td is None:
                raise Refused("the file carries no per-frame time dataset")
        else:
            raise BadTrajectory(
                f"{os.path.basename(path)} holds neither /coordinates (mdtraj, AMBER NetCDF-4) nor a "
                "/particles/*/position group (H5MD), so it is not a trajectory this reads"
            )
        if td.dtype.kind != "f":
            raise Refused(
                "the time dataset holds integers, not floating-point picoseconds"
            )
        units = td.attrs.get("units", td.attrs.get("unit", ""))
        if isinstance(units, bytes):
            units = units.decode("latin-1")
        if not isinstance(units, str):
            units = str(units)
        if not time_scale(units)[1]:
            raise Refused(f"the time dataset is in {units!r}, not picoseconds")
        if (
            td.compression is not None
            or td.shuffle
            or td.scaleoffset is not None
            or td.fletcher32
        ):
            raise Refused(
                "the time dataset is stored behind a compression filter, so its values cannot be "
                "rewritten in place"
            )
        return f, td
    except Exception:
        f.close()
        raise


# --------------------------------------------------
# Writing frames back out.


def write_piece(
    dest: str, run: Run, frames: int, pieces: list[tuple[str, int, int]]
) -> None:
    """Write a trajectory of the given frames: the head of run, with a DCD's
    frame count set to them, then the byte ranges in pieces, each
    (path, from, to), copied as they are. Nothing is left behind on failure."""

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    try:
        with open(dest, "wb") as out:
            if run.format == "dcd" and run.dcd is not None:
                head = bytearray(run.head)
                struct.pack_into(run.dcd.order + "i", head, DCD_NSET_OFFSET, frames)
                out.write(head)
            for path, a, b in pieces:
                with open(path, "rb") as src:
                    src.seek(a)
                    left = b - a
                    while left > 0:
                        chunk = src.read(min(left, 8 << 20))
                        if not chunk:
                            raise BadTrajectory(
                                f"{os.path.basename(path)} ended before byte {b}"
                            )
                        out.write(chunk)
                        left -= len(chunk)
            out.flush()
            os.fsync(out.fileno())
    except Exception:
        if os.path.exists(dest):
            os.remove(dest)
        raise


def joinable(a: Run, b: Run) -> None:
    """Whether b can follow a in one file."""

    if a.format != b.format:
        raise BadTrajectory(
            f"{a.name} is a {a.format} and {b.name} a {b.format}; segments joined into one file must be one format"
        )
    if a.format != "dcd":
        return
    x, y = a.dcd, b.dcd
    assert x is not None and y is not None
    if (x.natoms, x.crystal, x.four_dim, x.order, x.frame_bytes) != (
        y.natoms,
        y.crystal,
        y.four_dim,
        y.order,
        y.frame_bytes,
    ):
        raise BadTrajectory(
            f"{a.name} and {b.name} are DCDs laid out differently (atoms, unit cell or byte order), "
            "so one cannot follow the other in one file"
        )
