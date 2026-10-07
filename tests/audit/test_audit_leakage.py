"""Proves Tier 2 audit flags a synthetic look-ahead feature.

Each test feeds an inline source-code string to classify_features() (the
classifier walker exposed by experiments.audit.audit_leakage) and asserts
the verdict + evidence match the textbook failure mode:

    test_classifier_flags_negative_shift_as_leaking
        df.shift(-1) MUST be flagged Leaking with evidence=negative_shift.

    test_classifier_flags_center_true_rolling_as_leaking
        rolling(center=True) MUST be flagged Leaking with evidence=
        rolling_center_true.

    test_classifier_marks_normal_rolling_as_suspicious
        Trailing rolling (center=False / default) is suspicious-but-not-
        leaking; manual confirm is required.

    test_audit_leakage_catches_inflated_independence
        Alias delegating to the negative-shift test for VALIDATION.md
        row-naming compatibility ("MUST be flagged" wording).
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6)
# as pair-programming assistant. All design decisions and interpretations are
# the authors'.
from experiments.audit.audit_leakage import classify_features


def test_classifier_flags_negative_shift_as_leaking():
    src = '''
def f(df):
    result = df.copy()
    # PURE LEAK: uses df.shift(-1) which is one bar in the future.
    result["leaky_feature"] = df["spread"].shift(-1)
    return result
'''
    findings = classify_features(src)
    leak = [f for f in findings if f["feature"] == "leaky_feature"]
    assert len(leak) == 1, "should classify exactly one feature"
    assert leak[0]["verdict"] == "Leaking", f"got {leak[0]['verdict']}"
    assert "negative_shift" in leak[0]["evidence"]


def test_classifier_flags_center_true_rolling_as_leaking():
    src = '''
def f(df):
    result = df.copy()
    result["centered"] = df["spread"].rolling(3, center=True).mean()
    return result
'''
    findings = classify_features(src)
    assert findings[0]["verdict"] == "Leaking"
    assert "rolling_center_true" in findings[0]["evidence"]


def test_classifier_marks_normal_rolling_as_suspicious():
    """Trailing rolling is suspicious-but-not-leaking; manual confirm needed."""
    src = '''
def f(df):
    result = df.copy()
    result["normal"] = df.groupby("pid")["spread"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    return result
'''
    findings = classify_features(src)
    assert findings[0]["verdict"] == "Suspicious"


# Alias matching VALIDATION.md row 18-03-XX naming convention
# ("synthetic look-ahead feature MUST be flagged").
def test_audit_leakage_catches_inflated_independence():
    """Alias delegating to negative-shift test for VALIDATION.md naming compatibility."""
    test_classifier_flags_negative_shift_as_leaking()
