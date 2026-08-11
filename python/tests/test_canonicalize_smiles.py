"""Tests for canonicalize_smiles.py

**Rewritten 2026-08-11.** The previous version of this file tested an interface
that was deliberately deleted on 2026-07-17 (`b3352aa`): a script that took a
TOML file path, rewrote it in place, printed "Changed N SMILES", made `.bak`
backups, and "repaired" unescaped backslashes. Every test passed a TOML path as
argv, so the current script read the path itself as a SMILES and rejected it --
six failures, one cause, and they had been failing ever since.

Making them pass was never the goal. In-place rewriting *was the bug*: the
rewritten bytes no longer matched `mdrepo-submission.completed.json`, so the
manifest check failed on every ticket with ligands. Backslash repair was
explicitly decided against (2026-07-16) in favour of rejecting invalid TOML and
reporting the parse error. Those tests asserted behaviour that must not come
back, so they are gone; the chemistry assertions were ported.

The script is now a pure filter: N SMILES in argv -> N canonical lines on
stdout, in input order, touching no file.

These tests drive the CLI rather than importing canonicalize(), because the CLI
*is* the contract. mdr-process (validate.rs) spawns it with the SMILES as
positional arguments, requires stdout to have exactly one line per input, and
substitutes the results back into the ligand list **by position** with .zip().
It verifies the count and bails on a mismatch -- but it cannot detect a
reordering, which would silently attach one ligand's canonical SMILES to
another ligand's record. Hence test_batch_preserves_input_order.
"""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "canonicalize_smiles.py"


# --------------------------------------------------
def run(*smiles, cwd=None) -> subprocess.CompletedProcess:
    """Invoke the script the way mdr-process does: SMILES as positional args"""

    return subprocess.run(
        [sys.executable, str(SCRIPT), *smiles],
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
    )


# --------------------------------------------------
def lines(result: subprocess.CompletedProcess) -> list[str]:
    return result.stdout.splitlines()


# ==================================================
# The contract mdr-process depends on
# ==================================================


# --------------------------------------------------
def test_batch_returns_one_line_per_input():
    """N SMILES in, N lines out

    validate.rs bails with "returned N line(s) for M SMILES" otherwise, which
    fails the whole directory.
    """

    result = run("CCO", "c1ccccc1", "CC(=O)O")

    assert result.returncode == 0, result.stderr
    assert len(lines(result)) == 3


# --------------------------------------------------
def test_batch_preserves_input_order():
    """The dangerous one, and the reason this file exists

    validate.rs zips the output onto the ligand list, so position IS identity.
    A reordering would attach one ligand's canonical SMILES to another ligand's
    record -- wrong chemistry, no error, nothing downstream to catch it. The
    count check in validate.rs cannot see this.

    The expected forms are deliberately neither sorted nor reverse-sorted, so
    an accidental sort would fail rather than coincidentally pass.
    """

    result = run("OCC", "c1ccccc1", "CC(=O)O", "[N+H3]")

    assert result.returncode == 0, result.stderr
    assert lines(result) == ["CCO", "c1ccccc1", "CC(=O)O", "[NH3+]"]
    assert lines(result) != sorted(lines(result)), "a sort must be detectable here"


# --------------------------------------------------
def test_no_arguments_succeeds_silently():
    """A ligand-free directory must not be an error

    validate.rs returns early when there are no ligands so this should not
    arise in practice, but exiting non-zero on an empty batch would turn a
    harmless call into a failed directory.
    """

    result = run()

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


# --------------------------------------------------
def test_invalid_smiles_exits_nonzero_naming_the_offender():
    """The message is submitter-facing: it must say which SMILES was bad

    validate.rs surfaces stderr verbatim, so this text reaches a human who has
    to find the entry in their TOML.
    """

    result = run("not-a-smiles")

    assert result.returncode != 0
    assert "Invalid SMILES" in result.stderr
    assert "not-a-smiles" in result.stderr


# --------------------------------------------------
def test_one_bad_smiles_fails_the_whole_batch_and_emits_nothing():
    """No partial output, and the offender named is the bad one

    A partial list would be worse than none: mdr-process would either bail on
    the count or, if the counts happened to line up, substitute silently
    shifted values.
    """

    result = run("CCO", "not-a-smiles", "c1ccccc1")

    assert result.returncode != 0
    assert "not-a-smiles" in result.stderr
    assert result.stdout.strip() == "", "must not emit a partial batch"


# --------------------------------------------------
def test_touches_no_files(tmp_path):
    """The entire point of the 2026-07-17 purity work, pinned

    The old script rewrote mdrepo-metadata.toml in place, so the bytes stopped
    matching mdrepo-submission.completed.json and ticket::check_manifest failed
    every ticket with ligands. A second pass, or any --skip-download re-run,
    could not succeed. Nothing else in this suite would notice if that came
    back.
    """

    meta = tmp_path / "mdrepo-metadata.toml"
    original = '[[ligands]]\nsmiles = "OCC"\n'
    meta.write_text(original)
    before = sorted(p.name for p in tmp_path.iterdir())

    result = run("OCC", cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    assert meta.read_text() == original, "the metadata file must be untouched"
    assert sorted(p.name for p in tmp_path.iterdir()) == before, "no .bak, no new files"


# ==================================================
# Chemistry, ported from the previous file
# ==================================================


# --------------------------------------------------
def test_already_canonical_is_unchanged():
    result = run("CCO")

    assert result.returncode == 0, result.stderr
    assert lines(result) == ["CCO"]


# --------------------------------------------------
@pytest.mark.parametrize(
    "spellings,canonical",
    [
        (["OCC", "CCO"], "CCO"),
        (["C1CCCCC1", "C1CCCC(C1)"], "C1CCCCC1"),
    ],
    ids=["ethanol", "cyclohexane"],
)
def test_equivalent_spellings_converge(spellings, canonical):
    """Different ways of writing one molecule must produce identical output

    This is what makes the stored SMILES usable as a dedup/join key, which is
    why Design A ("canonical everywhere") was chosen over canonicalizing only
    on rejection.
    """

    result = run(*spellings)

    assert result.returncode == 0, result.stderr
    assert lines(result) == [canonical] * len(spellings)


# --------------------------------------------------
def test_nonstandard_protonation_is_normalised():
    """[N+H3] -> [NH3+], the load-bearing case

    This is the *only* stated reason canonicalization runs before validation:
    purr (the Rust validator) rejects [N+H3] and accepts [NH3+]. If OpenBabel
    stopped normalising it, ligands that process fine today would start failing
    validation, and nothing else here would explain why.
    """

    result = run("[N+H3]")

    assert result.returncode == 0, result.stderr
    assert lines(result) == ["[NH3+]"]


# --------------------------------------------------
def test_stereo_smiles_survives_canonicalization():
    """Backslash stereo bonds are real data -- ~2.4% of md_ligand rows

    Note what is NOT tested here any more: the old file asserted the script
    "repaired" a literal backslash in a double-quoted TOML string. That was
    deliberately removed. Backslashes are a TOML *quoting* problem, not a
    chemistry one ('C/C=C\\C' single-quoted parses fine), and the decision was
    to reject invalid TOML with its parse error rather than rewrite a
    submitter's file.
    """

    smiles = r"CC1(C)[NH+]=C2N(C1)C(=CS2)CS/C(=[NH+]\C1CCCCC1)/NC1CCCCC1"

    result = run(smiles)

    assert result.returncode == 0, result.stderr
    assert len(lines(result)) == 1
    assert "\\" in lines(result)[0], "stereo information must survive"
