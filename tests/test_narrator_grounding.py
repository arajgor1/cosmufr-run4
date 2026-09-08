"""
The narrator may not state a number the measurements did not supply.

This is the only thing standing between a language model and a plausible wrong
figure on a page whose entire argument is that every number on it was measured.
It is tested without touching the network: the check is a pure function, and its
job is to reject, so the cases that matter are the ones it must not let through.
"""
import pytest

from cosmufr.narrate import _grounded, _numbers, run_facts

SOURCE = (
    "Om: (how much matter the universe holds) predicted 0.32008, "
    "true value 0.31691, difference +0.00317\n"
    "w0: (what dark energy is doing) predicted -0.94110\n"
    "Ob: (how much ordinary matter there is) predicted 0.05330\n"
    "how much of this spectrum lies outside the range: 84 percent"
)


def test_rejects_a_number_nobody_measured():
    ok, why = _grounded("Matter density came back at 0.4123.", SOURCE)
    assert not ok
    assert "0.4123" in why


def test_rejects_a_rounded_measurement():
    """An approximation of a measurement is a figure nobody measured."""
    ok, _ = _grounded("Dark energy is about -0.94.", SOURCE)
    assert not ok


def test_accepts_a_supplied_number():
    ok, _ = _grounded("Matter density came back at 0.32008.", SOURCE)
    assert ok


def test_accepts_a_typographic_minus():
    """U+2212 is the same measurement, written with nicer punctuation."""
    ok, _ = _grounded("Dark energy came back at \u22120.94110.", SOURCE)
    assert ok


def test_accepts_dropping_a_sign_to_say_how_far_off():
    ok, _ = _grounded("Matter density was off by 0.00317.", SOURCE)
    assert ok


def test_accepts_counting_words():
    ok, _ = _grounded("Two of the eight came back well; 84 percent was outside.",
                      SOURCE)
    assert ok


def test_trailing_zeros_are_the_same_measurement():
    ok, _ = _grounded("Ordinary matter came back at 0.0533.", SOURCE)
    assert ok


def test_facts_name_every_parameter_in_words():
    """
    The narrator once read `mv` as a matter-to-radiation ratio because nothing
    told it otherwise. Every slot must arrive already named.
    """
    labels = ["Om", "s8", "h", "ns", "Ob", "w0", "mv", "wa"]
    facts = run_facts([0.3] * 8, None, labels)
    assert "combined mass of neutrinos" in facts["mv"]
    assert str(len(labels)) in str(facts["how many parameters it returns"])


def test_numbers_survives_junk():
    assert _numbers("no digits here") == []
