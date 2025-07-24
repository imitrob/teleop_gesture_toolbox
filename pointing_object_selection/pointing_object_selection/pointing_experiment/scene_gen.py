import numpy as np
from copy import deepcopy
from scene_getter.scene_lib.scene import Scene
from scene_getter.scene_lib.scene_object import SceneObject
# -----------------------------------------------------------------------------
# Constant object pools per scene complexity level
# -----------------------------------------------------------------------------
OBJECT_POOL_C1 = [
    "small blue plastic cup", 
    # "medium red plastic cup", 
    "medium red metal bowl",
    # "medium yellow plastic banana", 
    # "medium red plastic apple", 
    # "small brown plastic capsule",
    "small paper box", 
    # "small green plastic cube", 
    # "medium red plastic cube",
]

OBJECT_POOL_C2 = [
    ["small blue plastic cup", "medium red plastic cup"],
    ["small green plastic cube", "medium red plastic cube"],
]

OBJECT_POOL_C3 = [
    "medium yellow plastic banana", "medium red plastic apple", "small paper box",
]

# -----------------------------------------------------------------------------
# Helper classes & utilities
# -----------------------------------------------------------------------------
class Loc:
    """Categorises a point into a coarse region relative to an origin."""
    center_margin = 0.01 # [m]

    def __init__(self, xyz):
        self.x, self.y, self.z = xyz

    # ────────────────────────────────────────────────────────────────────────
    # Region helpers
    # ────────────────────────────────────────────────────────────────────────
    def region(self):
        if self.y < -self.center_margin:
            horiz = "left"
        elif self.y > self.center_margin:
            horiz = "right"
        else:
            horiz = "center"

        if self.x > self.center_margin:
            vert = "front"
        elif self.x < -self.center_margin:
            vert = "back"
        else:
            vert = "center"

        return f"{horiz}_{vert}"

    # Equality based on region only
    def __eq__(self, other):
        return isinstance(other, Loc) and self.region() == other.region()

    def __repr__(self):  # pragma: no cover
        return f"Loc(region={self.region()}, x={self.x:.2f}, y={self.y:.2f})"

    @property
    def annotation(self):
        r = self.region()
        if r == "center_center": return "center"
        return r.replace("_", " ") 

# ─────────────────────────────────────────────────────────────────────────────
# Region‑based search helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_locs(target, locs, offset, *, similar: bool):
    """Return index of first (dis)similar location wrt *target* region."""
    tgt_region = Loc(np.array(target) - np.array(offset)).region()
    valid_locs = []
    for i, loc in enumerate(locs):
        same = Loc(np.array(loc) - np.array(offset)).region() == tgt_region
        if same is similar:
            valid_locs.append(i)
    if len(valid_locs) > 0:
        return valid_locs
    raise ValueError("Suitable location not found.")

def find_near_dissimilar_loc(target, locs, offset):
    valid_locs = _find_locs(target, locs, offset, similar=False)
    return get_loc_near_id(target, valid_locs)

def find_near_similar_loc(target, locs, offset):
    valid_locs = _find_locs(target, locs, offset, similar=True)
    return get_loc_near_id(target, valid_locs)

def get_loc_near_id(loc, locs):
    dists = []
    for l in locs:
        dists.append(np.linalg.norm(l - loc))
    return np.argmin(dists)

# -----------------------------------------------------------------------------
# Core scene generator
# -----------------------------------------------------------------------------

def generate_random_scene(
    num_scene_objects: int,
    scene_complexity: int = 1,
    *,  # keyword‑only from this point
    ngon: int = 6,
    rings: int = 2,
    dist_base: float = 0.15,
    offset: np.ndarray = np.array([0.5, 0.0, 0.04]),
):
    """Return (Scene, object_locations, offset) for a randomised tabletop scene.

    * ``scene_complexity``:
        1. All objects unique (uses ``OBJECT_POOL_C1``)
        2. Contains one group of visually similar items (``OBJECT_POOL_C2``)
        3. Two identical targets plus distractors (``OBJECT_POOL_C3``)
    """

    # ------------------------------------------------------------------
    # 1. Generate candidate locations in concentric rounds around *offset*
    # ------------------------------------------------------------------
    locs = [offset]
    for ring in range(1, rings + 1):
        radius = ring * dist_base
        step_angle = 10 * ngon / ring  # angle granularity shrinks outward
        for deg in np.arange(0, 360, step_angle):
            locs.append(offset + np.array(
                    [np.sin(np.deg2rad(deg)) * radius, np.cos(np.deg2rad(deg)) * radius, 0.0]
                )
            )
    locs = np.array(locs)

    # ------------------------------------------------------------------
    # 2. Select and place objects according to complexity level
    # ------------------------------------------------------------------
    names: list[str] = []
    positions: list[np.ndarray] = []

    if scene_complexity == 1:
        name_pool = deepcopy(OBJECT_POOL_C1)
        locs_pool = list(deepcopy(locs))

        names.append( name_pool.pop(np.random.choice(len(name_pool))) )
        positions.append( locs_pool.pop(np.random.choice(range(len(locs_pool)))) )
        target_obj_loc = positions[0]

        remaining_objects = num_scene_objects - 1
        for i in range(remaining_objects):
            positions.append( locs_pool.pop(get_loc_near_id(target_obj_loc, locs_pool)) )
            names.append( name_pool.pop(np.random.choice(range(len(name_pool)))) )
        
    elif scene_complexity == 2:
        name_pool_groups = deepcopy(OBJECT_POOL_C2)
        locs_pool = list(deepcopy(locs))

        tgt_group = name_pool_groups.pop(np.random.randint(len(name_pool_groups)))
        names.append( tgt_group.pop(np.random.randint(len(tgt_group))) )
        positions.append( locs_pool.pop(np.random.choice(range(1, len(locs_pool)))) ) # range from 1 -> locs pool without the center
        target_obj_loc = positions[0]

        # place remaining in‑group objects in different region(s)
        for item in tgt_group:
            names.append(item)
            positions.append(locs_pool.pop(find_near_dissimilar_loc(positions[0], locs_pool, offset)))

        # fill the rest randomly from other groups
        remaining_objects = num_scene_objects - len(names)
        flat_rest = [o for g in name_pool_groups for o in g]
        while remaining_objects > 0:
            names.append( flat_rest.pop(np.random.choice(range(len(flat_rest)))) )
            positions.append( locs_pool.pop(get_loc_near_id(target_obj_loc, locs_pool)) )
            remaining_objects -= 1

    else:  # scene_complexity == 3
        name_pool = deepcopy(OBJECT_POOL_C3)
        locs_pool = list(deepcopy(locs))

        # special locations pool is without locations where y==0, these doesn't have similar location
        #   this is specific for ngon: int = 6 and rounds: int = 2
        special_pool = list(range(1, len(locs_pool)))
        special_pool.remove(16)
        special_pool.remove(10)

        names.append( name_pool.pop(np.random.choice(len(name_pool))) )
        positions.append( locs_pool.pop(np.random.choice(special_pool)) )
        target_obj_loc = positions[0]
        # put second identical targets in same region
        names.append(names[0])
        positions.append(locs_pool.pop(find_near_similar_loc(target_obj_loc, locs_pool, offset)))

        # add distractors
        remaining_objects = num_scene_objects - 2
        while remaining_objects > 0:
            names.append( name_pool.pop(np.random.choice(range(len(name_pool)))) )
            positions.append( locs_pool.pop(get_loc_near_id(target_obj_loc, locs_pool)) )
            remaining_objects -= 1

    # ------------------------------------------------------------------
    # 3. Build textual descriptions ("located at …")
    # ------------------------------------------------------------------
    annotated: list[str] = []
    for nm, pos in zip(names, positions):
        annotated.append(f"{nm} {Loc(pos - offset).annotation}")


    # ------------------------------------------------------------------
    # 4. Assemble Scene objects (expects Scene & SceneObject available)
    # ------------------------------------------------------------------
    scene_objects = []
    target_used = False
    target_obj_name = names[0]
    for idx, text in enumerate(annotated):
        obj_class = text.split()[-3]  # penultimate word
        name = "target_object" if not target_used and target_obj_name in text else f"{obj_class}{idx}"
        if name == "target_object":
            target_used = True
        scene_objects.append(
            SceneObject.from_dict(name, {"position": positions[idx], "params": text})
        )

    assert target_used, "No target object placed!"
    return Scene(name="target_object_scene", objects=scene_objects), locs, offset

if __name__ == "__main__":
    n = 300
    DEBUG = False

    for o in range(1, 6):
        for i in range(n):
            s, locs, of = generate_random_scene(o, 1)
            s.print_table_scene(locs, of)
            if DEBUG: input()

    for o in range(2, 5):
        for i in range(n):
            s, locs, of = generate_random_scene(o, 2)
            s.print_table_scene(locs, of)
            if DEBUG: input()

    for o in range(2, 5):
        for i in range(n):
            s, locs, of = generate_random_scene(o, 3)
            print(s)
            s.print_table_scene(locs, of)
            if DEBUG: input()