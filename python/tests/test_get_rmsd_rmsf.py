"""Tests for get_rmsd_rmsf.py

The ceiling check and the decimation, and the interaction between them.

Ticket 2339 prompted these tests. `GW4_NCOA2-c.nc` carried 20 consecutive
corrupt frames -- 45001 to 45020 of 80,000, coordinates and box and time all
written as zeros by the submitter's own cpptraj, which our fit then sent to
the XTC integer limit. RMSF rejected it at 588,039, and rightly.

The check was taking its max AFTER decimating to ~1,000 points, so it only
ever looked at one value in `num // 1000`. On 2339 that did not matter: RMSF
is per-CA and 251 atoms are under the cutoff, so nothing was decimated, and
RMSD's own max is 24.4 full against 24.0 decimated. RMSD missed the
corruption for an unrelated and unfixable reason -- rms.RMSD superimposes, so
a frame whose atoms all sit at one saturated coordinate centres to the origin
and scores 18.2, which is just a protein moving.

So the order is not what saved 2339. It is what would save the case 2339
makes obvious: RMSF decimates on any protein over 1,000 residues, RMSD on any
run over 1,000 frames, and a spike narrow enough to fall between strides
passes a check that never looks at it -- storing a clean-looking curve on top
of a damaged trajectory.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import get_rmsd_rmsf as g  # noqa: E402


# --------------------------------------------------
def test_normal_values_pass():
    """Real values from a healthy trajectory are well under the bar

    Measured, not invented: sims 98340, 98342 and 98343 are the clean -b, -d
    and -e replicates of ticket 2339's system, and their stored maxima are
    RMSD 24.40 / 24.14 / 24.36 and RMSF 38.65 / 39.07 / 39.49. So the headroom
    to 500 is about a factor of 12 on RMSF, not the two orders of magnitude a
    smaller system would suggest -- worth knowing before anyone lowers the
    ceiling. Note the RMSF is not small in absolute terms: nothing aligns the
    trajectory before rms.RMSF, so global drift is in the number.
    """

    g.check_ceiling("RMSF", g.MAX_RMSF, [12.4, 38.65, 39.49, 21.0])
    g.check_ceiling("RMSD", g.MAX_RMSD, [0.0, 18.2, 24.40, 22.1])


# --------------------------------------------------
def test_empty_values_are_refused():
    """No values at all is a failure, not a pass"""

    with pytest.raises(SystemExit) as excinfo:
        g.check_ceiling("RMSF", g.MAX_RMSF, [])

    assert "Failed to get RMSF values!" in str(excinfo.value)


# --------------------------------------------------
def test_value_over_the_ceiling_is_refused():
    """The message carries the value and the bar, verbatim as before

    mdr-process puts this string in the ticket log and in
    md_process_job.last_error, so its shape is a contract with whoever reads
    the failure.
    """

    with pytest.raises(SystemExit) as excinfo:
        g.check_ceiling("RMSF", g.MAX_RMSF, [2.1, 588039.7267995253, 1.9])

    assert str(excinfo.value) == (
        "Trajectory RMSF val 588039.7267995253 greater than 500"
    )


# --------------------------------------------------
def test_the_ceiling_is_exclusive():
    """Exactly 500 passes; the check is `>`, not `>=`"""

    g.check_ceiling("RMSF", g.MAX_RMSF, [500])

    with pytest.raises(SystemExit):
        g.check_ceiling("RMSF", g.MAX_RMSF, [500.0001])


# --------------------------------------------------
def test_sample_strides_past_a_narrow_spike():
    """The decimation really does discard the frames that matter

    Not a test of the check -- a test of the premise. Ticket 2339's numbers:
    80,000 values, corruption at 45001-45020, stride 80.
    """

    vals = [1.0] * 80000
    for i in range(45001, 45021):
        vals[i] = 2.1474e7

    sampled = g.sample(vals)

    assert len(sampled) == 1000
    assert max(sampled) == 1.0, "a spike off the stride survived sampling"
    assert max(vals) == 2.1474e7


# --------------------------------------------------
def test_spike_off_the_stride_still_trips_the_ceiling():
    """The regression: the check must see the full vector

    Run against the decimated list the same values pass. This is the shape of
    2339's corruption -- 20 frames at the XTC limit, 45001-45020 of 80,000 --
    used here as a spike generator, not as a claim about what 2339's RMSD did.
    """

    vals = [1.0] * 80000
    for i in range(45001, 45021):
        vals[i] = 2.1474e7

    g.check_ceiling("RMSD", g.MAX_RMSD, g.sample(vals))

    with pytest.raises(SystemExit) as excinfo:
        g.check_ceiling("RMSD", g.MAX_RMSD, vals)

    assert "greater than 500" in str(excinfo.value)


# --------------------------------------------------
def test_a_single_bad_frame_is_caught():
    """One corrupt frame in 80,000 had a 1-in-80 chance of being sampled

    The general case. Frame 45001 is not a multiple of 80, so the old order
    would not have looked at it; the check has to be indifferent to where the
    spike falls.
    """

    for spike_at in (0, 1, 79, 80, 45001, 79999):
        vals = [1.0] * 80000
        vals[spike_at] = 1e6

        with pytest.raises(SystemExit):
            g.check_ceiling("RMSD", g.MAX_RMSD, vals)
