from __future__ import annotations

from dataclasses import asdict, astuple, dataclass
from math import atan2, degrees, pi
from typing import ClassVar, Tuple

from micromind.geometry.vector import Vector2


@dataclass
class Line2:
    pt1: Vector2
    pt2: Vector2
    ZERO: ClassVar[Line2]
    UP: ClassVar[Line2]
    DOWN: ClassVar[Line2]
    LEFT: ClassVar[Line2]
    RIGHT: ClassVar[Line2]

    @staticmethod
    def from_point(
        point: Vector2, angle: float, size: float, centered: bool = False
    ) -> Line2:
        angle_rad = radians(angle)
        angle_rad_inv = radians(angle - 180)
        if centered:
            size = size / 2
            point_1 = point + Vector2(cos(angle_rad_inv), sin(angle_rad_inv)) * size
        else:
            point_1 = point
        point_2 = point + Vector2(cos(angle_rad), sin(angle_rad)) * size
        return Line2(point_1, point_2)

    def as_int_tuple(self):
        return self.pt1.as_int_tuple(), self.pt2.as_int_tuple()


Line2.ZERO = Line2(Vector2.ZERO, Vector2.ZERO)
Line2.UP = Line2(Vector2.ZERO, Vector2.UP)
Line2.DOWN = Line2(Vector2.ZERO, Vector2.DOWN)
Line2.LEFT = Line2(Vector2.ZERO, Vector2.LEFT)
Line2.RIGHT = Line2(Vector2.ZERO, Vector2.RIGHT)
