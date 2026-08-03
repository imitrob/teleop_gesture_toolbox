#!/usr/bin/env python
"""Gesture Link Lab: click gestures, see which action your links file names.

    python -m gesture_meaning.link_game            # http://127.0.0.1:8078

Standalone on purpose -- no ROS, no sensor, no detector, no robot. Load any
<user>_links.yaml and press the gesture buttons: the page asks this server what
the links mean, and the answer comes from the same OneToOneMapping the merger
uses, so what you see here is what the robot would do. Useful for checking a
links file before wiring up hardware, and for feeling how much work a
combination is to perform.

The verdict is always map_stamped's; the "needs ..." hints are worked out from
what it left over, so a hint can never disagree with the outcome.
"""
import argparse
import glob
import http.server
import json
import os
import urllib.parse

import yaml

from gesture_meaning.gesture_icons import GESTURE_ICONS
from gesture_meaning.one_to_one_mapping import (DEFAULT_LINKS, OneToOneMapping,
                                                derive_actions)

PAGE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "link_game.html")


def available_links() -> dict:
    """label -> path of the links files the page can offer with one button: the
    links.yaml shipped here, plus every links/<user>_links.yaml of hri_manager
    when that package is installed."""
    found = {"links.yaml": DEFAULT_LINKS}
    try:
        import hri_manager
    except ImportError:
        return found
    for path in sorted(glob.glob(f"{hri_manager.package_path}/links/*_links.yaml")):
        found[os.path.basename(path).removesuffix("_links.yaml")] = path
    return found


def game_state(links_yaml: str, clicks: list) -> dict:
    """What the page needs to draw itself, for one links file and one sequence
    of gesture clicks."""
    links = yaml.safe_load(links_yaml) or {}
    # Whatever file the user picked -- say what is wrong with it rather than
    # letting the mapping trip over it.
    if not isinstance(links, dict):
        return {"error": "The file is not a links yaml (expected a mapping at the top)."}
    if not isinstance(links.get("links", {}), dict):
        return {"error": "`links:` must be a mapping of link name -> link, not "
                         f"{type(links['links']).__name__}."}
    # The file carries the arity lists, not the vocabulary they add up to.
    links.setdefault("actions", derive_actions(links))
    mapping = OneToOneMapping(links, quiet=True)
    linked = sorted({gesture for gestures, _ in mapping.combinations for gesture in gestures})
    clicks = [str(click).lower() for click in clicks]

    # The real code path: clicks are a word stream, one stamp per click.
    fired = [action for _, action in mapping.map_stamped(
        [[index, gesture] for index, gesture in enumerate(clicks)], known_gestures=linked)]

    # Which clicks each fired action used, so what is left over is exact.
    leftover, triggered = list(clicks), []
    for action in fired:
        for gestures, candidate in mapping.combinations:
            if candidate == action and all(leftover.count(g) >= gestures.count(g)
                                           for g in set(gestures)):
                for gesture in gestures:
                    leftover.remove(gesture)
                triggered.append({"action": action, "gestures": list(gestures)})
                break

    pending = []
    for gestures, action in mapping.combinations:
        shown = [g for g in gestures if g in leftover]
        missing = [g for g in gestures if g not in leftover]
        if shown and missing:
            pending.append({"action": action, "shown": shown, "missing": missing})

    return {
        "user": links.get("user", ""),
        "gestures": linked,
        "actions": list(mapping.actions),
        "combinations": [{"gestures": list(g), "action": a} for g, a in mapping.combinations],
        "triggered": triggered,
        "pending": pending,
        "leftover": leftover,
        "gesture_icons": GESTURE_ICONS,
    }


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        path, _, query = self.path.partition("?")
        if path == "/api/links":
            offered = available_links()
            name = urllib.parse.parse_qs(query).get("name", [""])[0]
            if not name:
                return self._send(json.dumps(list(offered)), "application/json")
            # Only ever a file this server itself offered: the name is a key of
            # that list, never a path the page can steer.
            if name not in offered:
                return self.send_error(404, "No such links file")
            with open(offered[name]) as f:
                return self._send(f.read(), "text/plain")
        if path in ("/", "/index.html"):
            with open(PAGE) as f:
                return self._send(f.read(), "text/html")
        self.send_error(404)

    def do_POST(self):
        if not self.path.startswith("/api/state"):
            return self.send_error(404)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        try:
            state = game_state(body.get("links", ""), body.get("clicks", []))
        except yaml.YAMLError as e:
            state = {"error": f"Cannot read the yaml: {e}"}
        self._send(json.dumps(state), "application/json")

    def _send(self, text, content_type):
        payload = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass  # one line per click is noise


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8078)
    args = parser.parse_args()
    # Localhost only: it reads whatever yaml the page hands it.
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Gesture Link Lab on http://127.0.0.1:{args.port}  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("")


if __name__ == "__main__":
    main()
