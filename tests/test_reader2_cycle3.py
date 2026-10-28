"""Final-round reader rules (after the third corpus sample): ambiguous unit tokens, composite titles, capital-X
exponent offsets, thousands multiplier and superscript powers of ten."""
from afmquant.ocr import canonical_unit, parse_offset, parse_number_unit, parse_unit
from afmquant.reader2 import _k_multiplier, _pow10_labels


def test_ambiguous_unit_tokens_are_unknown():
    """'uu' is produced for 'nm' as often as for 'µm' in small print and 'mm' for 'nm': neither is a unit."""
    assert canonical_unit("uu") is None and canonical_unit("UU") is None and canonical_unit("(uu)") is None
    assert canonical_unit("mm") is None
    assert canonical_unit("um") == "µm" and canonical_unit("nm") == "nm" and canonical_unit("(nm)") == "nm"
    assert canonical_unit("ym") == "µm"                                 # a confusion that is unambiguous stays mapped
    assert canonical_unit("NA") is None and canonical_unit("nA") == "nA" and canonical_unit("na") == "nA"


def test_composite_or_logarithmic_title_yields_no_unit():
    assert canonical_unit("kcounts/80 ms") is None
    assert canonical_unit("Log(E/MPa)") is None
    assert canonical_unit(parse_unit("Height (nm)")) == "nm"       # titles are reduced by parse_unit first
    assert canonical_unit(parse_unit("Amplitude (pm)")) == "pm"
    assert canonical_unit("80 nm") == "nm"                          # a label with a glued unit keeps its unit


def test_offset_with_capital_x():
    assert parse_offset("X10-8") == 1e-8 and parse_offset("x10^-3") == 1e-3 and parse_offset("1e-9") == 1e-9
    assert parse_offset("1.4") is None


def test_thousands_multiplier_on_every_label():
    labs = [{"text": "21k", "num": 21.0}, {"text": "15k", "num": 15.0}, {"text": "3k", "num": 3.0}]
    assert _k_multiplier(labs)
    assert not _k_multiplier([{"text": "21k", "num": 21.0}, {"text": "15", "num": 15.0}])
    assert not _k_multiplier([{"text": "21k", "num": 21.0}])
    assert parse_number_unit("21k")[0] == 21.0                           # the multiplier is applied by the reader, not the parser


def test_superscript_powers_of_ten_are_a_log_scale():
    col = [{"text": "102", "num": 102.0, "kind": "col"}, {"text": "100", "num": 100.0, "kind": "col"},
           {"text": "20.0", "num": 20.0, "kind": "end"}]
    assert _pow10_labels(col)
    assert not _pow10_labels([{"text": "100", "num": 100.0, "kind": "col"}, {"text": "200", "num": 200.0, "kind": "col"}])
    assert not _pow10_labels([{"text": "100", "num": 100.0, "kind": "col"}, {"text": "100", "num": 100.0, "kind": "col"}])
