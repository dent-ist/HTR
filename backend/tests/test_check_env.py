import check_env as ce


def test_pascal_with_sm60_is_compatible():
    assert ce.arch_support((6, 1), ["sm_50", "sm_60", "sm_70"])[0] == ce.OK


def test_native_match():
    assert ce.arch_support((6, 1), ["sm_61"])[0] == ce.OK


def test_no_usable_kernels_warns():
    assert ce.arch_support((6, 1), ["sm_75", "sm_80", "sm_90"])[0] == ce.WARN


def test_ptx_only_warns():
    assert ce.arch_support((6, 1), ["compute_60"])[0] == ce.WARN


def test_disk_check_returns_status():
    assert ce.check_disk()[0] in {ce.OK, ce.WARN, ce.FAIL}
