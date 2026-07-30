#!/usr/bin/env python
"""The link game's one function: links yaml + gesture clicks -> what to draw.

No server and no browser: game_state is pure, so the rules the page shows can be
checked here. The point of the game is that its verdict is the mapping's, so the
cases that matter are the ones where a hint must agree with the outcome.
"""
import json
import threading
import urllib.error
import urllib.parse
import urllib.request

import pytest

from gesture_meaning.link_game import Handler, available_links, game_state

LINKS = """
user: tester
single_object_actions: [pick, touch]
zero_object_actions: [stop]
links:
  l1:
    action_template: pick
    action_gestures: [[grab, swipe_up]]
  l2:
    action_template: stop
    action_gestures: [[five]]
"""


def test_a_complete_combination_fires():
    state = game_state(LINKS, ["grab", "swipe_up"])
    assert state["triggered"] == [{"action": "pick", "gestures": ["grab", "swipe_up"]}]
    assert state["leftover"] == [] and state["pending"] == []


def test_half_a_combination_waits_and_says_what_is_missing():
    state = game_state(LINKS, ["grab"])
    assert state["triggered"] == []
    assert state["leftover"] == ["grab"]
    assert state["pending"] == [{"action": "pick", "shown": ["grab"], "missing": ["swipe_up"]}]


def test_a_hint_cannot_disagree_with_the_outcome():
    """Everything pending is a gesture left over, and nothing left over was
    counted towards a fired action."""
    state = game_state(LINKS, ["grab", "five", "grab", "swipe_up"])
    assert [t["action"] for t in state["triggered"]] == ["pick", "stop"]
    assert state["leftover"] == ["grab"]
    for entry in state["pending"]:
        assert set(entry["shown"]) <= set(state["leftover"])


def test_only_linked_gestures_are_offered():
    """The buttons are the gestures the file gives meaning to -- a file that
    links nothing is playable but can fire nothing."""
    assert game_state(LINKS, [])["gestures"] == ["five", "grab", "swipe_up"]
    assert game_state("user: nobody\nlinks: {}\n", [])["gestures"] == []


def test_the_vocabulary_is_reported_whole():
    """Including actions no gesture reaches, so a links file missing a link is
    visible as an action that can never fire."""
    assert game_state(LINKS, [])["actions"] == ["stop", "pick", "touch"]


def test_a_file_that_is_not_links_yaml_is_reported_not_raised():
    assert "error" in game_state("- just\n- a list\n", [])
    assert "error" in game_state("links: [not, a, mapping]\n", [])
    # An empty file is a links file with nothing linked, not a broken one.
    assert game_state("", []) == game_state("links: {}\n", [])


def test_the_shipped_links_are_playable():
    from gesture_meaning.one_to_one_mapping import DEFAULT_LINKS
    with open(DEFAULT_LINKS) as f:
        shipped = f.read()
    state = game_state(shipped, [])
    assert state["combinations"], "the shipped links.yaml can fire nothing"
    first = state["combinations"][0]
    assert game_state(shipped, first["gestures"])["triggered"] == [first]


# The server around it #

@pytest.fixture
def server():
    import http.server
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_every_links_file_of_the_workspace_is_offered():
    """The shipped one always, each user's file when hri_manager is installed."""
    offered = available_links()
    assert "links.yaml" in offered
    for name, path in offered.items():
        assert game_state(open(path).read(), [])["gestures"] is not None, name
    try:
        import hri_manager
    except ImportError:
        pytest.skip("hri_manager not installed, only the shipped file is offered")
    assert "demo" in offered, offered
    assert offered["demo"].endswith("links/demo_links.yaml")


def test_a_links_file_the_server_never_offered_is_refused(server):
    """The button sends a name, not a path: nothing else is readable."""
    for name in ("../../../etc/passwd", "/etc/passwd", "nope"):
        with pytest.raises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(f"{server}/api/links?name={urllib.parse.quote(name, safe='')}")
        assert raised.value.code == 404


def test_the_page_and_the_endpoints_answer(server):
    assert "Gesture Link Lab" in urllib.request.urlopen(f"{server}/").read().decode()
    names = json.loads(urllib.request.urlopen(f"{server}/api/links").read())
    assert "links.yaml" in names
    assert "links" in urllib.request.urlopen(
        f"{server}/api/links?name=links.yaml").read().decode()

    request = urllib.request.Request(
        f"{server}/api/state", method="POST",
        data=json.dumps({"links": LINKS, "clicks": ["five"]}).encode(),
        headers={"Content-Type": "application/json"})
    state = json.loads(urllib.request.urlopen(request).read())
    assert state["triggered"] == [{"action": "stop", "gestures": ["five"]}]
