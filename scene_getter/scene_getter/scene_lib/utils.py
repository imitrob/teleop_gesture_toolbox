
from scene_getter.scene_lib.scene_object import SceneObject

class cc:
    H = '\033[95m'
    OK = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    W = '\033[93m'
    F = '\033[91m'
    E = '\033[0m'
    B = '\033[1m'
    U = '\033[4m'

def print_table_scene(scene,
                      locs,
                      offset,
                      scene2 = None,
                      x_bounds=(-0.31, 1.06),
                      y_bounds=(-0.40, 0.40),
                      cols=60,
                      rows=20,
                      print_empty_symbol = " ",
                      do_printing = False,
                      ):
    """
    Visualise a table scene in plain ASCII.

    Parameters
    ----------
    objects : list[tuple[str, float, float]]
        Sequence of (name, x, y) in metres.
    x_bounds, y_bounds : tuple[float, float]
        Min & max coordinates of the table along X and Y.
    cols, rows : int
        Resolution of the ASCII grid (more = finer).

    Example
    -------
    >>> objects = [('cube', 0.10, 0.10),
    ...            ('red drawer', -0.25, -0.10),
    ...            ('paper box', 0.95, 0.00)]
    >>> print_table_scene(objects)
    """
    for loc in locs:
        if list(loc) not in [list(p) for p in scene.object_positions]:
            scene.objects.append( SceneObject.from_dict("dummy", {"position": loc, "params": "dummy" }) )

    ROBOT_NAME = "Robot Franka Emika Panda"
    SENSOR_NAME = "Leap Motion Sensor"
    object_list = []
    for o in scene.objects:
        object_list.append([f"{o.name},\t{o.params}", o.position[0], o.position[1]])
    object_list.extend([[ROBOT_NAME, 0.0, 0.0], [SENSOR_NAME, 1.06, 0.4]])
    
    object_list2 = []
    if scene2 is not None:
        for o in scene2.objects:
            object_list2.append([f"{o.name},\t{o.params}", o.position[0], o.position[1]])
    
    

    xmin, xmax = x_bounds
    ymin, ymax = y_bounds
    sx = (cols - 1) / (xmax - xmin)      # metres → column index
    sy = (rows - 1) / (ymax - ymin)      # metres → row index

    # Blank grid filled with dots
    grid = [[print_empty_symbol for _ in range(cols)] for _ in range(rows)]

    
    for i in range(len(grid)):
        grid[i][int(round((offset[0] - xmin) * sx))] = "|"
    for i in range(len(grid[0])):
        grid[int(round((ymax - offset[1]) * sy))][i] = "-"
    


    symbols = []
    for idx, (name, x, y) in enumerate(object_list, 1):
        # Clamp off-table points
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            print(f"WARNING: {name!r} at ({x:.2f},{y:.2f}) "
                             "is outside the table bounds")
            continue
        c = int(round((x - xmin) * sx))
        r = int(round((ymax - y) * sy))   # invert Y so +Y is up
        
        if name == "dummy,\tdummy":
            symbol = "+"
        elif name[0].upper() == "O":
            symbol = str(idx)
        elif name in [ROBOT_NAME, SENSOR_NAME]:
            symbol = name[0].upper()
        else:
            symbol = str(idx)
        # Avoid collisions by falling back to index if needed
        if grid[r][c] not in [print_empty_symbol, "|", "-"]:
            symbol = str(idx)
        grid[r][c] = symbol

        if name != "dummy,\tdummy":
            symbols.append((symbol, name))

    for idx, (name, x, y) in enumerate(object_list2, 1):
        symbol = name[0].upper()
        for s_, n_ in symbols:
            if name[:-2] in n_: # if "bowl" in "medium red metal bowl center"
                symbol = s_
                break

        c = int(round((x - xmin) * sx))
        r = int(round((ymax - y) * sy))   # invert Y so +Y is up
        try:
            grid[r][c] = f"{cc.W}{symbol}{cc.E}"
        except IndexError:
            pass

    # Frame the table with a border
    border_h = '+' + '-' * cols + '+'
    final_s = ""
    final_s += border_h + "\n"
    # print(border_h)
    grid.reverse()

    grid[3].insert(0, "| /|\\LEFT")
    grid[len(grid)//2-1].insert(0, "| |CENTER")
    grid[-3].insert(0, "| \\|/RIGHT")

    for row in grid:
        row.reverse()
        # print('|' + ''.join(row) + '|')
        final_s += '|' + ''.join(row) + '|' + "\n"
    FRONT = 10; BACK = -20
    border_h = border_h[:FRONT] + "FRONT" + border_h[FRONT+5:BACK-4] + "BACK" + border_h[BACK:]
    final_s += border_h + "\n"
    # print(border_h)

    # Legend
    # print("\nLegend:")
    final_s += "\nLegend:\n"
    for symbol, name in symbols:
        # print(f"  {symbol} – {name}")
        final_s += f"  {symbol} – {name}" + "\n"

    if do_printing:
        print(final_s)
    return final_s