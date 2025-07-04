

from pointing_object_selection.deictic_lib import DeiticLib


def test_deictic():
    dl = DeiticLib()
    # test 1
    line_points = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    test_points = ((0.0, 0.0, 1.00000), (10.0, 0.0, 1.0), (10.0, 0.0, 2.0))
    id, dist, _ = dl.get_id_of_closest_point_to_line(line_points, test_points)
    assert (id, dist) == (0, 1.0)

    # test 2
    line_points = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    test_points = ((0.0, 0.0, 0.99999), (10.0, 0.0, 1.0), (10.0, 0.0, 2.0))
    id, dist, _ = dl.get_id_of_closest_point_to_line(line_points, test_points)
    assert (id, dist) == (0, 0.99999)

    # test 3
    line_points = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    test_points = ((0.0, 0.0, 0.99999), (10.0, 0.0, 1.0), (10.0, 0.0, 2.0))
    id, dist, _ = dl.get_id_of_closest_point_to_line(line_points, test_points, max_dist=0.3)
    assert (id, dist) == (None, 0.99999)

if __name__ == "__main__":
    test_deictic()