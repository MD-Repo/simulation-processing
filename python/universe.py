"""
Author : Ken Youens-Clark <kyclark@gmail.com>
Date   : 2026-11-07
Purpose: Open a release's files as an MDAnalysis Universe the way every
         command here reads them

mda.Universe(topology, *coordinates), except that a CHARMM-GUI "Chamber"
.parm7 is read through ParmEd. A Chamber topology is AMBER's format holding a
CHARMM force field (its FORCE_FIELD_TYPE says so) and opens with %FLAG CTITLE
where AMBER's own has %FLAG TITLE. MDAnalysis's TOPParser refuses it
outright; ParmEd reads it, and MDAnalysis builds a Universe from the ParmEd
structure. zenodo:6487373 is eleven such systems, each a .parm7 and its .nc
runs with no coordinate file.
"""

import warnings


# --------------------------------------------------
def is_chamber(topology: str) -> bool:
    """Whether the topology is a CHARMM-GUI Chamber .parm7."""

    try:
        with open(topology, "rb") as fh:
            return b"%FLAG CTITLE" in fh.read(4096)
    except OSError:
        return False


# --------------------------------------------------
def open_universe(topology: str, *coordinates: str, **kwargs):
    """The Universe of the topology and coordinates. Keyword arguments go to
    mda.Universe and are not used for a Chamber topology, which ParmEd
    reads whatever its extension."""

    warnings.filterwarnings("ignore")
    import MDAnalysis as mda

    if is_chamber(topology):
        import parmed

        return mda.Universe(parmed.load_file(topology), *coordinates)
    return mda.Universe(topology, *coordinates, **kwargs)
