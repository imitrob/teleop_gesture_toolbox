
from pointing_object_selection.pointing_experiment.scene_gen import generate_random_scene

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
    s.print_table_scene(locs, of)
    s, locs, of = generate_random_scene(3, 1)
    s.print_table_scene(locs, of)
    s, locs, of = generate_random_scene(3, 1)
    s.print_table_scene(locs, of)
    s, locs, of = generate_random_scene(3, 1)
    s.print_table_scene(locs, of)

if __name__ == "__main__":
    s, locs, of = generate_random_scene(3, 1)
    s.print_table_scene(locs, of)