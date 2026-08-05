import gesture_sentence_maker.gesture_processor as gesture_processor


def test_hri_manager_mode_uses_editable_user_file(monkeypatch):
    monkeypatch.setattr(gesture_processor, "HRI_MANAGER_AVAILABLE", True)

    assert gesture_processor._links_capability("casper") == (
        True, "casper_links.yaml")


def test_hri_manager_mode_requires_a_user(monkeypatch):
    monkeypatch.setattr(gesture_processor, "HRI_MANAGER_AVAILABLE", True)

    assert gesture_processor._links_capability("") == (
        False, "<user>_links.yaml")


def test_standalone_mode_uses_read_only_default_file(monkeypatch):
    monkeypatch.setattr(gesture_processor, "HRI_MANAGER_AVAILABLE", False)

    editable, source = gesture_processor._links_capability("ignored-user")

    assert not editable
    assert source == "links.yaml"


def test_standalone_mode_loads_default_even_without_user(monkeypatch):
    expected = {"user": "default", "links": {}}
    monkeypatch.setattr(gesture_processor, "HRI_MANAGER_AVAILABLE", False)
    monkeypatch.setattr(
        gesture_processor, "load_links", lambda name_user: expected)

    assert gesture_processor._user_settings("") is expected
