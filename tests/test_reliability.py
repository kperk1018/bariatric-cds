from src.reliability import tier, gate
from src.risk import assess_preop_risk


def test_tbwl_tiers_match_s5():
    assert tier("TBWL", 2) == "green"
    assert tier("TBWL", 3) == "green"
    assert tier("TBWL", 1) == "amber"
    assert tier("TBWL", 6) == "red"


def test_red_years_block_point_predictions():
    assert gate("TBWL", 6)["allow_point_prediction"] is False
    assert gate("FML", 5)["allow_point_prediction"] is False
    assert gate("TBWL", 2)["allow_point_prediction"] is True


def test_preop_threshold_flag():
    assert assess_preop_risk(7.8)["below_threshold"] is True
    assert assess_preop_risk(12.0)["below_threshold"] is False
    assert assess_preop_risk(7.8)["flag"] == "at_risk"
