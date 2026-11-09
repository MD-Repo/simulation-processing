"""
Author : Nick Eddy <eddy.nicholas@gmail.com>
Date   : 2026-11-06
Purpose: Read a Desmond -out.cms (Schrodinger's Maestro format) into the
         system's particles, bonds and cell, in msys's particle order

A finished Desmond run leaves an -out.cms, the system in Maestro format, and
a _trj directory of .dtr frames. MDAnalysis reads neither. The .cms is read
here the way DESRES's msys reads it, because Desmond reads its .cms through
msys and writes its trajectory in msys's particle order, so a converter has
to build the same order to map the frames onto the atoms.

The reader follows DEShawResearch/msys src/mae/mae.cxx, commit 30aa5b6: the
same tokens, the same grammar, the same typed values. A field named
i_m_residue_number is an integer keyed m_residue_number; "<>" is null; a
table's rows are counted as read, whatever its header declares.

The system is built as msys builds it (src/mae/import_mae.cxx, src/import.cxx,
src/system.cxx at 30aa5b6):

  * the full_system block is skipped, and the system assembled from the
    component blocks in file order;
  * each component contributes its m_atom rows, then its ffio_pseudo rows,
    the virtual sites, whose atomic number is 0;
  * a chain is one (component, chain name, segment name); a residue is one
    (chain, number, name, insertion code), wherever in the file its atoms
    are; names are trimmed; a missing residue name is UNK;
  * bonds come from each component's m_bond, lower index first, repeats
    dropped.

The order is not the order of a component's ffio_sites template. For TIP4P
water the template is O H H M, but msys puts every atom first and every M
after; a converter that followed the template put M sites 1.8 nm from their
oxygens, where msys's order puts all 11,959 of them 0.015 nm away.

Ported from mdrepo-agent's internal/desmond (mae.go, system.go) when the
conversion moved here (mdrepo-agent#98). The agent keeps its own reader for
counting a .cms's atoms; this one feeds desmond_convert.py.
"""

from dataclasses import dataclass, field
from typing import Optional

Box = tuple[tuple[float, float, float], ...]

ZERO_BOX: Box = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


class MaestroError(ValueError):
    """Text that is not a Maestro file this reader understands."""


class AlchemicalError(MaestroError):
    """An FEP file: msys rewrites alchemical stages before it reads them, and
    this reader does not."""


# --------------------------------------------------
# Tokens and values


@dataclass(frozen=True)
class Token:
    text: str
    quoted: bool = False


_SEPARATORS = set(' \t\n\r\f\v[]{}"')


def tokenize(s: str) -> list[Token]:
    """Split Maestro text as msys's tokenizer does. Whitespace separates;
    [ ] { } stand alone; "..." is a string in which a backslash makes the
    next character literal; a # at the start of a token opens a comment that
    runs to the end of the line or the next #."""

    out: list[Token] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c in " \t\n\r\f\v":
            i += 1
        elif c == "#":
            i += 1
            while i < n and s[i] != "\n" and s[i] != "#":
                i += 1
            i += 1
        elif c in "[]{}":
            out.append(Token(c))
            i += 1
        elif c == '"':
            i += 1
            buf = []
            while i < n and s[i] != '"':
                if s[i] == "\\" and i + 1 < n:
                    i += 1
                buf.append(s[i])
                i += 1
            i += 1
            out.append(Token("".join(buf), quoted=True))
        else:
            j = i
            while j < n and s[j] not in _SEPARATORS:
                j += 1
            out.append(Token(s[i:j]))
            i = j
    return out


@dataclass(frozen=True)
class Value:
    """One Maestro value, kept as text and read by the type its field
    declares, as msys reads it: integers with atoi, reals with atof."""

    text: str = ""
    null: bool = False

    def str(self, default: str = "") -> str:
        return default if self.null else self.text

    def int(self, default: int = 0) -> int:
        """Like C's atoi: the leading integer, or 0."""
        if self.null:
            return default
        s = self.text.strip()
        end = 0
        if end < len(s) and s[end] in "+-":
            end += 1
        while end < len(s) and s[end].isdigit():
            end += 1
        try:
            return int(s[:end])
        except ValueError:
            return 0

    def float(self, default: float = 0.0) -> float:
        """Like C's atof: the leading number, or 0."""
        if self.null:
            return default
        s = self.text.strip()
        for end in range(len(s), 0, -1):
            try:
                return float(s[:end])
            except ValueError:
                continue
        return 0.0


NULL = Value(null=True)


# --------------------------------------------------
# Blocks and tables


@dataclass
class Table:
    """An indexed block, name[n] { ... }, read by column."""

    cols: dict[str, list[Value]] = field(default_factory=dict)
    rows: int = 0

    def at(self, col: str, row: int) -> Value:
        c = self.cols.get(col)
        if c is None or row >= len(c):
            return NULL
        return c[row]


@dataclass
class Block:
    """A Maestro block: its own values, its tables and its sub-blocks."""

    values: dict[str, Value] = field(default_factory=dict)
    tables: dict[str, Table] = field(default_factory=dict)
    blocks: dict[str, "Block"] = field(default_factory=dict)


class _Parser:
    def __init__(self, toks: list[Token]):
        self.toks = toks
        self.i = 0

    def peek(self) -> str:
        return self.toks[self.i].text if self.i < len(self.toks) else ""

    def done(self) -> bool:
        return self.i >= len(self.toks)

    def next(self) -> Token:
        if self.done():
            raise MaestroError("unexpected end of file")
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, want: str) -> None:
        try:
            t = self.next()
        except MaestroError as e:
            raise MaestroError(f"expected {want!r}: {e}")
        if t.text != want or t.quoted:
            raise MaestroError(f"expected {want!r}, got {t.text!r} at token {self.i}")

    def schema(self) -> list[tuple[str, str]]:
        """Field names up to ":::", split into (kind, key)."""
        fields = []
        while self.peek() != ":::":
            n = self.next().text
            if len(n) < 2 or n[1] != "_" or n[0] not in "birs":
                raise MaestroError(f"line predicted a schema, but {n!r} is not a field")
            fields.append((n[0], n[2:]))
        self.expect(":::")
        return fields

    def val(self) -> Value:
        t = self.next()
        if t.text == "<>" and not t.quoted:
            return NULL
        return Value(t.text)

    def block_body(self) -> Block:
        """ "{ schema ::: values sub-blocks }"."""
        self.expect("{")
        b = Block()
        for _, key in self.schema():
            b.values[key] = self.val()
        while self.peek() != "}":
            if self.done():
                raise MaestroError("block not closed")
            self.sub_block(b)
        self.expect("}")
        return b

    def array_body(self) -> Table:
        """ "[ n ] { schema ::: rows ::: [one nested array] }". msys skips one
        level of nested array, with a warning, and so does this."""
        self.expect("[")
        self.next()  # the declared count
        self.expect("]")
        self.expect("{")
        fields = self.schema()
        t = Table()
        while self.peek() != ":::":
            self.next()  # the row index
            for _, key in fields:
                t.cols.setdefault(key, []).append(self.val())
            t.rows += 1
        self.expect(":::")
        nt = self.next()
        if nt.text != "}":
            self.array_body()
            self.expect("}")
        return t

    def sub_block(self, into: Block) -> None:
        t = self.next()
        c = t.text
        if not c or not (c[0].isalpha() or c[0] == "_"):
            raise MaestroError(f"predicted a block name, have {t.text!r}")
        try:
            if self.peek() == "[":
                into.tables[c] = self.array_body()
            else:
                into.blocks[c] = self.block_body()
        except MaestroError as e:
            raise MaestroError(f"{c}: {e}")


def parse_cts(text: str) -> list[Block]:
    """Parse a whole file: a nameless header block, then named blocks, of
    which the f_m_ct ones are the structures."""

    p = _Parser(tokenize(text))
    if p.peek() == "{":
        try:
            p.block_body()
        except MaestroError as e:
            raise MaestroError(f"header: {e}")
    cts = []
    while not p.done():
        t = p.next()
        try:
            if p.peek() == "[":
                p.array_body()
                continue
            b = p.block_body()
        except MaestroError as e:
            raise MaestroError(f"{t.text}: {e}")
        if t.text == "f_m_ct":
            cts.append(b)
    return cts


# --------------------------------------------------
# The system


@dataclass
class Particle:
    """One atom or virtual site."""

    component: int
    chain: str
    segid: str
    resid: int
    resname: str
    insertion: str
    name: str
    atomic_number: int  # 0 for a virtual site
    x: float  # angstroms
    y: float
    z: float
    pseudo: bool
    chain_index: int
    residue_index: int


@dataclass
class System:
    """A Desmond system in msys's particle order, which is the order of the
    particles in its trajectory."""

    particles: list[Particle] = field(default_factory=list)
    bonds: list[tuple[int, int]] = field(default_factory=list)  # lower first
    chains: int = 0
    residues: int = 0
    # The periodic cell's vectors in angstroms, from the last component that
    # states one, as msys takes it; zero if none does.
    box: Box = ZERO_BOX
    # The force field names the components' ffio_ff blocks carry, verbatim
    # and without repeats, in file order: "OPLS3e_REASSIGN", "S-OPLS".
    forcefields: list[str] = field(default_factory=list)

    def atoms(self) -> tuple[list[Particle], list[tuple[int, int]], list[int]]:
        """The structure without its virtual sites, as msys's structure-only
        load does: the particles with an atomic number above 0, and the bonds
        between them, renumbered. keep holds each atom's index in the full
        system, which is its index in the trajectory."""

        remap: dict[int, int] = {}
        atoms, keep = [], []
        for i, p in enumerate(self.particles):
            if p.atomic_number > 0:
                remap[i] = len(atoms)
                atoms.append(p)
                keep.append(i)
        bonds = [
            (remap[a], remap[b]) for a, b in self.bonds if a in remap and b in remap
        ]
        return atoms, bonds, keep


_CELL = (
    ("chorus_box_ax", "chorus_box_ay", "chorus_box_az"),
    ("chorus_box_bx", "chorus_box_by", "chorus_box_bz"),
    ("chorus_box_cx", "chorus_box_cy", "chorus_box_cz"),
)


def cell_of(ct: Block) -> Optional[Box]:
    """A component's periodic cell, as msys's import_cell reads it."""

    box = []
    for row in _CELL:
        vals = []
        for n in row:
            v = ct.values.get(n)
            if v is None:
                return None
            vals.append(v.float(0.0))
        box.append(tuple(vals))
    return tuple(box)  # type: ignore[return-value]


def parse(text: str) -> System:
    """Read the text of a .cms file."""

    cts = parse_cts(text)
    stages = set()
    for ct in cts:
        v = ct.values.get("fepio_stage")
        if v is not None:
            stages.add(v.int(0))
    if 1 in stages and 2 in stages:
        raise AlchemicalError(
            "alchemical (FEP) .cms, which msys rewrites before reading; not supported"
        )

    s = System()
    chains: dict[tuple[int, str, str], int] = {}
    residues: dict[tuple[int, int, str, str], int] = {}
    bonds: set[tuple[int, int]] = set()
    component = -1

    def add(chain, segid, resid, resname, name, insertion, anum, x, y, z, pseudo):
        chain, segid = chain.strip(), segid.strip()
        resname, name, insertion = resname.strip(), name.strip(), insertion.strip()
        ck = (component, chain, segid)
        ci = chains.setdefault(ck, len(chains))
        rk = (ci, resid, resname, insertion)
        ri = residues.setdefault(rk, len(residues))
        s.particles.append(
            Particle(component, chain, segid, resid, resname, insertion, name, anum,
                     x, y, z, pseudo, ci, ri)
        )  # fmt: skip

    for ct in cts:
        if ct.values.get("ffio_ct_type", NULL).str("") == "full_system":
            continue
        box = cell_of(ct)
        if box is not None:
            s.box = box
        atoms = ct.tables.get("m_atom")
        if atoms is None:
            continue
        component += 1
        base = len(s.particles)
        for j in range(atoms.rows):
            add(
                atoms.at("m_chain_name", j).str(""),
                atoms.at("m_pdb_segment_name", j).str(""),
                atoms.at("m_residue_number", j).int(0),
                atoms.at("m_pdb_residue_name", j).str("UNK"),
                atoms.at("m_pdb_atom_name", j).str(""),
                atoms.at("m_insertion_code", j).str(""),
                atoms.at("m_atomic_number", j).int(0),
                atoms.at("m_x_coord", j).float(0.0),
                atoms.at("m_y_coord", j).float(0.0),
                atoms.at("m_z_coord", j).float(0.0),
                False,
            )
        bt = ct.tables.get("m_bond")
        if bt is not None:
            for j in range(bt.rows):
                ai, aj = bt.at("m_from", j).int(0), bt.at("m_to", j).int(0)
                if ai < 1 or aj < 1 or ai > atoms.rows or aj > atoms.rows:
                    raise MaestroError(
                        f"bond {j + 1} between nonexistent atoms {ai}, {aj}"
                    )
                if ai == aj:
                    raise MaestroError(f"bond {j + 1} to self {ai}")
                if ai > aj:
                    ai, aj = aj, ai
                bonds.add((base + ai - 1, base + aj - 1))
        ff = ct.blocks.get("ffio_ff")
        if ff is not None:
            name = ff.values.get("ffio_name", NULL).str("").strip()
            if name and name not in s.forcefields:
                s.forcefields.append(name)
            ps = ff.tables.get("ffio_pseudo")
            if ps is not None:
                for j in range(ps.rows):
                    add(
                        ps.at("ffio_chain_name", j).str(""),
                        ps.at("ffio_pdb_segment_name", j).str(""),
                        ps.at("ffio_residue_number", j).int(0),
                        ps.at("ffio_pdb_residue_name", j).str("UNK"),
                        ps.at("ffio_atom_name", j).str(""),
                        "",
                        0,
                        ps.at("ffio_x_coord", j).float(0.0),
                        ps.at("ffio_y_coord", j).float(0.0),
                        ps.at("ffio_z_coord", j).float(0.0),
                        True,
                    )
    s.bonds = sorted(bonds)
    s.chains, s.residues = len(chains), len(residues)
    return s


def load(path: str) -> System:
    """Read a .cms file."""

    with open(path) as fh:
        text = fh.read()
    try:
        return parse(text)
    except MaestroError as e:
        raise type(e)(f"{path}: {e}")
