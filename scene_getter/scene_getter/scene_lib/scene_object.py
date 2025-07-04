# Pose = np.array([x,y,z,roll,pitch,yaw])
# Point = np.array([x,y,z]) or Vector3 np.array([xlen, ylen, zlen])

from __future__ import annotations
from typing import Iterable, Sequence
import numpy as np
from pydantic import BaseModel, Field, field_validator, ConfigDict


class SceneObject(BaseModel):
    """Static description of an object in the scene."""
    name: str = Field(..., description="Object name / identifier")
    position: np.ndarray = Field(..., description="3-element XYZ position (metres, world frame)")
    orientation: np.ndarray = Field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]),
        description="Quaternion [x, y, z, w]",
    )
    params: str = Field(default="", description="Optional free-form parameter string")
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    
    @field_validator("position", mode="before")
    @classmethod
    def _as_pos_array(cls, value: Iterable[float]) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if arr.shape != (3,):
            raise ValueError("`position` must have exactly 3 elements (x, y, z)")
        return arr

    @field_validator("orientation", mode="before")
    @classmethod
    def _as_quat_array(cls, value: Iterable[float]) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if arr.shape != (4,):
            raise ValueError("`orientation` must have exactly 4 elements (x, y, z, w)")
        return arr

    @property
    def type(self) -> str:
        return self.__class__.__name__

    @property
    def quaternion(self) -> np.ndarray:
        return self.orientation

    @property
    def info(self):
        print(self.__str__())

    def __str__(self) -> str:  # noqa: Dunder
        pos = np.round(self.position, 2)
        return f"{self.name},\t{self.type},\t{pos},\t{self.params}"

    @classmethod
    def from_dict(cls, name: str, data):
        """Mimic the original convenience constructor."""
        # If the caller passes only a position (tuple / list / array)
        if isinstance(data, (tuple, list, np.ndarray)):
            return cls(name=name, position=data)

        # Otherwise assume a full mapping with 'position' key (and optional others)
        return cls(name=name, **data)