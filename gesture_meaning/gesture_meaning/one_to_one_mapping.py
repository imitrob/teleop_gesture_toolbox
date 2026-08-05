"""Gesture -> action via user links.

A default would emit action words the user's own vocabulary may not contain, and
the reasoner is constrained to that vocabulary.

    links:
      link1:
        action_template: pick
        action_gestures: [[grab, swipe_up]]     # static + dynamic, BOTH needed
      link2:
        action_template: push
        action_gestures: [[swipe_left], [swipe_right]]   # either one is enough

One entry of `action_gestures` is a combination: every gesture in it has to be
shown for the action to be named, so `grab` alone means nothing above. Several
entries are alternatives.

Called in-process by whoever has the gestures: the sentence maker before it
publishes /modality/gestures, and every merge method on the words it received.
`python -m gesture_meaning.link_game` plays with a links file in a browser.
"""
import os

import yaml

DEFAULT_LINKS = os.path.join(os.path.dirname(os.path.realpath(__file__)), "links.yaml")


def derive_actions(links: dict) -> list:
    """The vocabulary, which is not stored in the yaml: it is the union of the
    arity lists (the same rule hri_manager.user_links applies)."""
    return (links.get("zero_object_actions") or []) \
        + (links.get("single_object_actions") or []) \
        + (links.get("double_object_actions") or []) \
        + (links.get("directional_actions") or [])


def load_links(name_user: str = "") -> dict:
    """A user's links, read from hri_manager when that package is installed.

    gesture_meaning is usable without the rest of the HRI stack, so with no
    hri_manager to ask, the links.yaml shipped beside this module is read
    instead -- same format, one file, no user. A missing user file inside
    hri_manager is still an error: falling back there would answer a mistyped
    --name_user with somebody else's vocabulary.
    """
    try:
        from hri_manager.user_links import load_user_links
        return load_user_links(name_user)
    except ImportError as e:
        print(f"hri_manager not installed ({e}), using {DEFAULT_LINKS}", flush=True)
        links = dict(yaml.safe_load(open(DEFAULT_LINKS)) or {})
        links["actions"] = derive_actions(links)
        return links


class OneToOneMapping:
    """A combination of gestures is mapped to one action command.

    Args:
        user_links: the dict from hri_manager.user_links.load_user_links, or
            just its "links" section. Empty means no gesture has meaning yet.
        quiet: suppress the note printed when a gesture is ignored.
    """

    def __init__(self, user_links: dict | None = None, quiet: bool = False):
        self.quiet = quiet
        self.combinations = []      # [(gestures, action)], gesture names lowercase
        self.gesture_actions = {}   # gesture -> action of the first combination naming it
        self._reported = set()
        user_links = user_links or {}
        # Either the whole links file or just its "links" section. A file whose
        # `links:` is absent or empty simply gives no gesture any meaning, so
        # entries that are not links are skipped rather than trusted.
        links = user_links.get("links") if "links" in user_links else user_links
        for name, link in (links if isinstance(links, dict) else {}).items():
            action = link.get("action_template", "") if isinstance(link, dict) else ""
            if not action:
                continue
            for entry in link.get("action_gestures", []):
                names = [entry] if isinstance(entry, str) else list(entry)
                gestures = tuple(dict.fromkeys(g.lower() for g in names))
                if not gestures:
                    continue
                self.combinations.append((gestures, action))
                for gesture in gestures:
                    first = self.gesture_actions.setdefault(gesture, action)
                    if first != action and not quiet:
                        print(f"Gesture {gesture!r} is linked to both {first!r} and "
                              f"{action!r} ({name}); using {first!r}", flush=True)
        # Longest combination first, so a two-gesture link is matched before a
        # one-gesture link that shares a gesture with it.
        self.combinations.sort(key=lambda combination: -len(combination[0]))

        # The action vocabulary the reasoner is constrained to. Given only a
        # links section, all that is known is the actions the links name.
        self.actions = (user_links or {}).get("actions") or list(
            dict.fromkeys(action for _, action in self.combinations))

    def _scores(self, gesture_names: list, gesture_probs: list) -> list:
        """[(score, gestures, action), ...], one per linked combination.

        A combination is scored by the product of its gestures' probabilities,
        which is what "all of them have to be shown" means for a detector that
        reports a distribution rather than words: half a combination scores 0
        however certain that half is.

        Taken as the geometric mean, so that combinations of different lengths
        stay comparable -- a plain product of two probabilities is always below
        either of them, which would make a one-gesture link win over a
        two-gesture one almost regardless of the evidence. With every
        combination the same length (a static + dynamic pair per link) the mean is a monotone transform of
        the product, so it does not change which action wins.
        """
        detected = dict(zip((g.lower() for g in gesture_names), gesture_probs))
        scored = []
        for gestures, action in self.combinations:
            product = 1.0
            for gesture in gestures:
                product *= float(detected.get(gesture, 0.0))
            scored.append((product ** (1 / len(gestures)), gestures, action))
        return scored

    def map_probs(self, gesture_names: list, gesture_probs: list):
        """(actions, action_probs) from one probability per detected gesture.
        An action reachable by several alternatives takes the best of them.

        `actions` is the user's whole vocabulary, so the probabilities can be
        merged with another modality's; actions no gesture reaches score 0.0.
        Everything at 0.0 means no gesture named an action -- the caller must
        not argmax that, since it would name the first action in the vocabulary.
        """
        best = {action: 0.0 for action in self.actions}
        for score, _, action in self._scores(gesture_names, gesture_probs):
            best[action] = max(best.get(action, 0.0), score)
        return list(best), [best[action] for action in best]

    def best_combination(self, gesture_names: list, gesture_probs: list):
        """(gestures, action) most likely to have been shown, or None when no
        linked combination was shown at all."""
        scored = self._scores(gesture_names, gesture_probs)
        if not scored:
            return None
        score, gestures, action = max(scored, key=lambda s: s[0])
        return (gestures, action) if score > 0.0 else None

    def gesture_to_action(self, gesture_name: str):
        """The action this gesture commands on its own. None when it is only
        part of a combination, or not a linked gesture at all (an unlinked
        gesture, or an object name)."""
        gesture = gesture_name.lower()
        return next((action for gestures, action in self.combinations
                     if gestures == (gesture,)), None)

    def map_stamped(self, stamped_words: list, known_gestures=None) -> list:
        """Convert a [[stamp, word], ...] list: every complete combination of
        gestures becomes its action word, and other words pass through
        unchanged. A combination is emitted at the stamp of its earliest
        gesture, since that is when the user began commanding it.

        Gestures left over -- unlinked, or one half of a combination whose other
        half never came -- carry no meaning, so they are dropped and named once.

        `known_gestures` is the gesture vocabulary of the detector, when the
        caller knows it. Without it a word that no link mentions cannot be told
        apart from an object name, so it passes through instead of being
        dropped.
        """
        known = {g.lower() for g in (known_gestures or [])}
        pool, out = {}, []   # pool: index -> (stamp, gesture); out: (index, stamp, word)
        for index, (stamp, word) in enumerate(stamped_words):
            if isinstance(word, str) and (word.lower() in self.gesture_actions
                                          or word.lower() in known):
                pool[index] = (stamp, word.lower())
            else:
                out.append((index, stamp, word))

        for gestures, action in self.combinations:
            while (picked := self._match(pool, gestures)) is not None:
                out.append((min(picked), min(pool[i][0] for i in picked), action))
                for i in picked:
                    del pool[i]

        for index in sorted(pool):
            self._report(pool[index][1])
        return [[stamp, word] for _, stamp, word in sorted(out, key=lambda t: t[0])]

    def _match(self, pool: dict, gestures: tuple):
        """Indices of one occurrence of every gesture of the combination, or
        None when the combination is not complete in the pool."""
        picked = []
        for gesture in gestures:
            found = next((i for i in sorted(pool)
                          if pool[i][1] == gesture and i not in picked), None)
            if found is None:
                return None
            picked.append(found)
        return picked

    def _report(self, gesture: str):
        if self.quiet or gesture in self._reported:
            return
        self._reported.add(gesture)
        partial = next((gestures for gestures, _ in self.combinations
                        if gesture in gestures), None)
        if partial:
            missing = [g for g in partial if g != gesture]
            print(f"Gesture {gesture!r} means {self.gesture_actions[gesture]!r} only "
                  f"shown together with {missing}, ignored on its own", flush=True)
        else:
            print(f"Gesture {gesture!r} has no link, ignored "
                  f"(add it inside the gesture dashboard)", flush=True)
