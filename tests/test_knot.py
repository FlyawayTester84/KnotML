from unknot_rl.utils.knot import generate_random_knot, is_unknot


def test_generate_random_knot():
    knot = generate_random_knot(crossings=7)
    assert len(knot.crossings) == 7


def test_is_unknot_behavior():
    knot = generate_random_knot(crossings=3)
    assert is_unknot(knot) is False

    empty = generate_random_knot(crossings=0)
    assert is_unknot(empty) is True


def test_reidemeister_I_removes_crossing():
    knot = generate_random_knot(crossings=3)
    before = len(knot.crossings)
    knot.reidemeister_I(0)
    assert len(knot.crossings) == max(0, before - 1)
