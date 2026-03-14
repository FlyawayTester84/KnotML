from unknot_rl.utils.knot import generate_random_knot, is_unknot


def test_generate_random_knot():
    knot = generate_random_knot(crossings=7)
    assert len(knot) == 7


def test_is_unknot_default_false():
    knot = generate_random_knot(crossings=3)
    assert is_unknot(knot) is False
