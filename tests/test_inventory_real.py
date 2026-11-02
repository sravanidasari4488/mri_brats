"""Modality and folder-name rules for the real-MRI inventory."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.inventory_real import folder_has_identifying_text, guess_modality, study_tier


def test_t1c_token_match():
    for desc in ("eeT1W_SPIR CE", "T1W_FFE CE", "T1W_SPIR_CE", "T1W_SPIR CE", "eT1W Gd", "T1 post", "T1W+C"):
        label, rule, usable = guess_modality(desc)
        assert label == "T1c", (desc, label, rule)
        assert usable is True
        assert "CE" in rule or "Gd" in rule or "post" in rule or "+C" in rule


def test_clear_is_not_contrast():
    label, _rule, usable = guess_modality("eT1W_IR_4mm CLEAR")
    assert label == "T1"
    assert usable is True
    label, _rule, _usable = guess_modality("eT2W_TSE CLEAR")
    assert label == "T2"


def test_ffe_without_t1_is_t2_star():
    label, rule, usable = guess_modality("FFE")
    assert label == "T2*"
    assert usable is False
    assert "FFE" in rule
    label, _rule, usable = guess_modality("T1W_FFE")
    assert label == "T1"
    assert usable is True


def test_excluded_not_brain_tumor_protocol():
    for desc in ("Survey", "SURV", "MobiView", "Spine T2", "BTFE", "STIR"):
        label, rule, usable = guess_modality(desc)
        assert label == "excluded", (desc, label)
        assert usable is False
        assert rule == "not brain tumor protocol"


def test_aux_series_recorded_not_input():
    cases = {
        "SWIp_fast": "SWI",
        "VSWIp_fast MINI MIP": "SWI",
        "MIP - VEN_3D_PCA": "VEN",
        "VEN_3D_PCA": "VEN",
        "s3DI_MC_HR": "MRA",
        "MIP - s3DI_MC_HR": "MRA",
    }
    for desc, want in cases.items():
        label, rule, usable = guess_modality(desc)
        assert label == want, (desc, label, rule)
        assert usable is False
        assert "not used as input" in rule


def test_standard_brain_inputs():
    assert guess_modality("eT2W_FLAIR SPIR CLEAR")[0] == "FLAIR"
    assert guess_modality("eFLAIR_SPIR")[0] == "FLAIR"
    assert guess_modality("eT1W_SE")[0] == "T1"
    assert guess_modality("sT1W_3D_HR")[0] == "T1"
    assert guess_modality("eT2W_TSE SENSE")[0] == "T2"


def test_tiers():
    full = {"FLAIR": True, "T1": True, "T1c": True, "T2": True}
    assert study_tier(full) == "FULL"
    assert study_tier({**full, "T1c": False}) == "NO_CONTRAST"
    assert study_tier({"FLAIR": True, "T1": False, "T1c": False, "T2": True}) == "FLAIR_T2_ONLY"
    assert study_tier({"FLAIR": True, "T1": True, "T1c": False, "T2": False}) == "UNUSABLE"
    assert study_tier({"FLAIR": False, "T1": True, "T1c": True, "T2": True}) == "UNUSABLE"


def test_identifying_folder_names_are_flagged_without_using_them():
    assert folder_has_identifying_text("CHINNAMMAI.PRATHIPATI   50Y F_MR_Study1")
    assert folder_has_identifying_text("MADHU U  29Y M_MR_Study1")
    assert not folder_has_identifying_text("26070216161267453698_MR_Study1")
    assert not folder_has_identifying_text("26070409540769354286_MR_Study1 2")


if __name__ == "__main__":
    test_t1c_token_match()
    test_clear_is_not_contrast()
    test_ffe_without_t1_is_t2_star()
    test_excluded_not_brain_tumor_protocol()
    test_aux_series_recorded_not_input()
    test_standard_brain_inputs()
    test_tiers()
    test_identifying_folder_names_are_flagged_without_using_them()
    print("inventory_real ok")
