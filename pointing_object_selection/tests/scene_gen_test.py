from pointing_experiment_selfcontined import generate_random_scene


def test_scene_generation():
    n = 1000
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