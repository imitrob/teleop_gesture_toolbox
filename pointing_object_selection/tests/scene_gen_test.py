
from pointing_object_selection.pointing_experiment.pointing_experiment import generate_random_scene
from pointing_object_selection.pointing_experiment.utils import print_table_scene


def test_scene_generation():
    n = 100
    DEBUG = False

    for o in range(1, 6):
        for i in range(n):
            s = generate_random_scene(o, 1)
            print(s)
            if DEBUG: input()

    for o in range(2, 5):
        for i in range(n):
            s = generate_random_scene(o, 2)
            print(s)
            if DEBUG: input()

    for o in range(2, 5):
        for i in range(n):
            s = generate_random_scene(o, 3)
            print(s)
            if DEBUG: input()

def test_scene_printing():
    s, locs, of = generate_random_scene(3, 1)
    print_table_scene(s, locs, of)
    s, locs, of = generate_random_scene(3, 1)
    print_table_scene(s, locs, of)
    s, locs, of = generate_random_scene(3, 1)
    print_table_scene(s, locs, of)
    s, locs, of = generate_random_scene(3, 1)
    print_table_scene(s, locs, of)
