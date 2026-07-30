#!/usr/bin/env python
"""OneToOneMapping on hand-built links and word streams.

Needs no ROS, no links file on disk and no detector: the links dict is written
inline, so every case states the whole contract it asserts. The mapping is what
stands between a raw gesture name and the reasoner's constrained vocabulary, so
the important cases here are the ones where a word must NOT come out: half a
combination, an unlinked gesture, and a link naming nothing.
"""
import pytest

from gesture_meaning.one_to_one_mapping import OneToOneMapping

# pick needs the static AND the dynamic gesture; push accepts either swipe.
LINKS = {"links": {
    "link1": {"action_template": "pick", "action_gestures": [["grab", "swipe_up"]]},
    "link2": {"action_template": "push", "action_gestures": [["swipe_left"],
                                                            ["swipe_right"]]},
}}
KNOWN = ["grab", "swipe_up", "swipe_left", "swipe_right", "point", "five", "two"]


@pytest.fixture
def mapping():
    return OneToOneMapping(LINKS, quiet=True)


# Building the mapping #

def test_a_combination_needs_every_gesture_of_the_entry(mapping):
    """Two gestures in one entry are a compound, not alternatives."""
    assert mapping.map_stamped([[0.0, "grab"], [0.2, "swipe_up"]],
                               known_gestures=KNOWN) == [[0.0, "pick"]]
    assert mapping.map_stamped([[0.0, "grab"]], known_gestures=KNOWN) == []
    assert mapping.map_stamped([[0.0, "swipe_up"]], known_gestures=KNOWN) == []
    assert mapping.gesture_to_action("grab") is None


def test_separate_entries_are_alternatives(mapping):
    assert mapping.gesture_to_action("swipe_left") == "push"
    assert mapping.gesture_to_action("swipe_right") == "push"
    assert mapping.map_stamped([[0.0, "swipe_right"]], known_gestures=KNOWN) == \
        [[0.0, "push"]]


def test_gesture_names_are_matched_case_insensitively():
    m = OneToOneMapping({"links": {"l": {"action_template": "pick",
                                         "action_gestures": [["Grab", "Swipe_Up"]]}}},
                        quiet=True)
    assert m.map_stamped([[0.0, "GRAB"], [0.1, "swipe_up"]]) == [[0.0, "pick"]]


def test_a_single_gesture_may_be_given_without_a_list():
    """action_gestures: [grab] and [[grab]] mean the same thing."""
    m = OneToOneMapping({"links": {"l": {"action_template": "stop",
                                         "action_gestures": ["grab"]}}}, quiet=True)
    assert m.gesture_to_action("grab") == "stop"


def test_the_links_section_may_be_passed_on_its_own():
    assert OneToOneMapping(LINKS["links"], quiet=True).combinations == \
           OneToOneMapping(LINKS, quiet=True).combinations


# Word streams #

def test_a_combination_is_emitted_at_its_earliest_gesture(mapping):
    """The action word replaces the whole combination and takes the stamp of its
    first gesture, because that is when the user began commanding it -- the
    merger interleaves it with the voice words by that stamp."""
    stamped = [[0.1, "grab"], [0.4, "cup1"], [0.6, "swipe_up"]]
    assert mapping.map_stamped(stamped, known_gestures=KNOWN) == \
        [[0.1, "pick"], [0.4, "cup1"]]


def test_gestures_of_a_combination_need_not_be_adjacent(mapping):
    """A pointed-at object may land between the two halves of one compound."""
    assert mapping.map_stamped([[0.0, "grab"], [0.1, "two"], [0.2, "swipe_up"]],
                               known_gestures=KNOWN) == [[0.0, "pick"]]


def test_a_repeated_combination_is_mapped_once_per_occurrence(mapping):
    stamped = [[0.0, "grab"], [0.1, "swipe_up"], [0.5, "grab"], [0.6, "swipe_up"]]
    assert mapping.map_stamped(stamped, known_gestures=KNOWN) == \
        [[0.0, "pick"], [0.5, "pick"]]


def test_half_a_combination_left_over_is_dropped(mapping):
    """Three grabs and one swipe_up complete one pick; the surplus grabs mean
    nothing on their own and must not reach the reasoner."""
    stamped = [[0.0, "grab"], [0.1, "grab"], [0.2, "swipe_up"], [0.3, "grab"]]
    assert mapping.map_stamped(stamped, known_gestures=KNOWN) == [[0.0, "pick"]]


def test_a_longer_combination_wins_over_a_shorter_one():
    """`grab` alone means touch, but with swipe_up it means pick: the compound is
    matched first, so the pair is not spent on the single-gesture link."""
    m = OneToOneMapping({"links": {
        "l1": {"action_template": "touch", "action_gestures": [["grab"]]},
        "l2": {"action_template": "pick", "action_gestures": [["grab", "swipe_up"]]},
    }}, quiet=True)
    assert m.map_stamped([[0.0, "grab"], [0.1, "swipe_up"]]) == [[0.0, "pick"]]
    assert m.map_stamped([[0.0, "grab"]]) == [[0.0, "touch"]]


def test_an_unlinked_gesture_is_dropped_when_the_vocabulary_is_known(mapping):
    """`five` is detectable but no link mentions it, so it carries no meaning and
    must not reach the reasoner as an object name."""
    assert mapping.map_stamped([[0.0, "five"], [0.2, "swipe_left"]],
                               known_gestures=KNOWN) == [[0.2, "push"]]


def test_without_the_vocabulary_an_unmentioned_word_passes_through(mapping):
    """An unlinked gesture cannot be told apart from an object name, so dropping
    it would silently eat `cup1`."""
    assert mapping.map_stamped([[0.0, "five"]]) == [[0.0, "five"]]


def test_a_non_string_candidate_passes_through_untouched(mapping):
    """Voice may deliver a ProbsVector rather than a word once STT reports
    n-best; the mapping is a gesture-name lookup and leaves it alone."""
    probs = {"cup1": 0.7, "cup2": 0.3}
    assert mapping.map_stamped([[0.0, probs]], known_gestures=KNOWN) == [[0.0, probs]]


def test_empty_input_maps_to_empty_output(mapping):
    assert mapping.map_stamped([]) == []


# No meaning without links #

def test_no_links_means_no_gesture_has_meaning():
    """There is deliberately no built-in mapping: a default would emit action
    words the user's own vocabulary may not contain."""
    m = OneToOneMapping({}, quiet=True)
    assert m.gesture_to_action("grab") is None
    assert m.map_stamped([[0.0, "grab"]], known_gestures=KNOWN) == []
    assert OneToOneMapping(None, quiet=True).combinations == []


def test_a_link_naming_no_action_is_ignored():
    m = OneToOneMapping({"links": {"l": {"action_gestures": [["grab"]]}}}, quiet=True)
    assert m.combinations == []


def test_a_links_file_with_no_links_section_gives_no_meaning():
    """A file may legitimately have no `links:` yet -- Part 3 of the setup writes
    the vocabulary, Part 5 the links -- and its other keys are not links."""
    m = OneToOneMapping({"objects": ["cube"], "single_object_actions": ["pick"]}, quiet=True)
    assert m.combinations == []
    assert m.map_stamped([[0.0, "grab"]], known_gestures=KNOWN) == []


# Notes on stdout #

def test_an_incomplete_combination_says_what_is_missing(capsys):
    OneToOneMapping(LINKS).map_stamped([[0.0, "grab"]], known_gestures=KNOWN)
    out = capsys.readouterr().out
    assert "swipe_up" in out, f"does not name the missing gesture:\n{out}"


def test_an_unlinked_gesture_is_named_once(capsys):
    m = OneToOneMapping(LINKS)
    m.map_stamped([[0.0, "five"], [0.1, "five"]], known_gestures=KNOWN)
    m.map_stamped([[0.2, "five"]], known_gestures=KNOWN)
    out = capsys.readouterr().out
    assert out.count("'five'") == 1, f"reported more than once:\n{out}"
    assert "link_gesture_to_action" in out, "the note does not say how to fix it"


def test_a_gesture_linked_twice_keeps_its_first_action_and_says_so(capsys):
    """Two links claiming one gesture is a configuration mistake; resolving it
    silently would make the meaning depend on dict order."""
    m = OneToOneMapping({"links": {
        "l1": {"action_template": "pick", "action_gestures": [["grab"]]},
        "l2": {"action_template": "push", "action_gestures": [["grab"]]},
    }})
    assert m.map_stamped([[0.0, "grab"]]) == [[0.0, "pick"]]
    out = capsys.readouterr().out
    assert "grab" in out and "push" in out and "pick" in out, out


def test_quiet_suppresses_the_notes(capsys):
    OneToOneMapping(LINKS, quiet=True).map_stamped([[0.0, "five"]], known_gestures=KNOWN)
    assert capsys.readouterr().out == ""


# Fallback links #

def test_links_come_from_hri_manager_when_it_is_installed():
    from gesture_meaning.one_to_one_mapping import load_links
    assert load_links("demo")["user"] == "demo"


def test_the_shipped_links_are_used_without_hri_manager(monkeypatch, capsys):
    """gesture_meaning is usable on its own, and a gesture has no meaning without
    a links file: with no hri_manager to ask, the one shipped here is read."""
    import builtins
    from gesture_meaning.one_to_one_mapping import DEFAULT_LINKS, load_links
    real_import = builtins.__import__

    def no_hri_manager(name, *args, **kwargs):
        if name.startswith("hri_manager"):
            raise ImportError("no hri_manager")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_hri_manager)
    links = load_links("demo")
    monkeypatch.undo()

    assert DEFAULT_LINKS in capsys.readouterr().out, "does not say which file it fell back to"
    assert links["user"] == "default"
    # `actions` is derived from the arity lists, as hri_manager derives it.
    assert set(links["actions"]) == {"pick", "touch", "put"}
    m = OneToOneMapping(links, quiet=True)
    assert m.map_stamped([[0.0, "grab"], [0.1, "swipe_up"]]) == [[0.0, "pick"]]
    for gesture, action in m.gesture_actions.items():
        assert action in links["actions"], f"{gesture} names {action}, not in the vocabulary"


def test_a_mistyped_user_is_still_an_error():
    """Falling back for a missing user file would answer a typo with somebody
    else's vocabulary."""
    from gesture_meaning.one_to_one_mapping import load_links
    with pytest.raises(FileNotFoundError):
        load_links("no_such_user")


# Probability vectors (what the detector reports) #

# Probabilities as the detector delivers them: one per gesture of Gs, static
# first then dynamic.
GS = ["grab", "swipe_left", "swipe_right", "swipe_up"]


def test_map_probs_scores_a_combination_by_its_gestures_together(mapping):
    actions, probs = mapping.map_probs(GS, [0.6, 0.1, 0.0, 0.5])
    scored = dict(zip(actions, probs))
    assert scored["pick"] == pytest.approx((0.6 * 0.5) ** 0.5)
    assert scored["push"] == pytest.approx(0.1)


def test_combinations_of_different_lengths_stay_comparable(mapping):
    """Both gestures of pick seen at 0.8, push's single gesture at 0.7: pick is
    the better reading. A plain product would score pick 0.64 and hand it to
    push."""
    gestures, action = mapping.best_combination(GS, [0.8, 0.7, 0.0, 0.8])
    assert action == "pick"
    assert gestures == ("grab", "swipe_up")


def test_map_probs_names_no_action_without_a_full_combination(mapping):
    """Certain about grab, saw no dynamic gesture at all: pick scores 0, so
    nothing is named. Publishing the argmax of this would name an action the
    user never commanded."""
    actions, probs = mapping.map_probs(GS, [1.0, 0.0, 0.0, 0.0])
    assert max(probs) == 0.0
    assert mapping.best_combination(GS, [1.0, 0.0, 0.0, 0.0]) is None


def test_map_probs_takes_the_best_of_two_alternatives(mapping):
    actions, probs = mapping.map_probs(GS, [0.0, 0.3, 0.8, 0.0])
    assert dict(zip(actions, probs))["push"] == 0.8


def test_best_combination_ignores_gestures_no_link_mentions(mapping):
    """A likely but unlinked gesture cannot outvote a linked combination."""
    m = OneToOneMapping(LINKS, quiet=True)
    gestures, action = m.best_combination(GS + ["five"], [0.4, 0.0, 0.0, 0.4, 0.99])
    assert action == "pick"


def test_map_probs_spans_the_whole_vocabulary():
    """The probabilities are merged with another modality's, so every action the
    user has needs an entry -- including the ones no gesture reaches."""
    links = dict(LINKS, actions=["pick", "push", "stop"])
    actions, probs = OneToOneMapping(links, quiet=True).map_probs(GS, [1.0, 0.0, 0.0, 1.0])
    assert actions == ["pick", "push", "stop"]
    assert dict(zip(actions, probs))["stop"] == 0.0
