import threading
from types import SimpleNamespace

import gesture_sentence_maker.gesture_processor as gesture_processor

GestureSentence = gesture_processor.GestureSentence


def _ctx(mtime):
    return SimpleNamespace(user="alex", _settings_mtime=mtime, user_settings={},
                           mapping=None, settings_lock=threading.RLock(),
                           _links_mtime=lambda: 2.0,
                           publish_meaning_info=lambda: None)


def test_reloads_when_the_file_changed(monkeypatch):
    monkeypatch.setattr(gesture_processor, "_user_settings",
                        lambda user: {"links": {}, "actions": ["pick"]})
    ctx = _ctx(1.0)
    GestureSentence.reload_settings_if_changed(ctx)
    assert ctx.user_settings["actions"] == ["pick"]
    assert ctx._settings_mtime == 2.0
    assert ctx.mapping is not None


def test_untouched_file_is_not_reread(monkeypatch):
    def fail(user):
        raise AssertionError("links file re-read although it did not change")

    monkeypatch.setattr(gesture_processor, "_user_settings", fail)
    GestureSentence.reload_settings_if_changed(_ctx(2.0))


def test_no_user_never_reloads(monkeypatch):
    def fail(user):
        raise AssertionError("reloaded without a user")

    monkeypatch.setattr(gesture_processor, "_user_settings", fail)
    ctx = _ctx(1.0)
    ctx.user, ctx._links_mtime = "", lambda: None
    GestureSentence.reload_settings_if_changed(ctx)
