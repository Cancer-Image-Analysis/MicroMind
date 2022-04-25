from __future__ import annotations

from dataclasses import asdict, astuple, dataclass
from math import atan2, degrees, pi
from typing import ClassVar, Tuple


@dataclass
class Vector2:
    x: float
    y: float
    ZERO: ClassVar[Vector2]
    UP: ClassVar[Vector2]
    DOWN: ClassVar[Vector2]
    LEFT: ClassVar[Vector2]
    RIGHT: ClassVar[Vector2]

    def __sub__(self, other: Vector2) -> Vector2:
        return Vector2(self.x - other.x, self.y - other.y)

    def __add__(self, other: Vector2) -> Vector2:
        return Vector2(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar: float) -> Vector2:
        return Vector2(self.x * scalar, self.y * scalar)

    def dot(self, other: Vector2) -> float:
        return self.x * other.x + self.y * other.y

    def norm(self) -> float:
        return self.dot(self) ** 0.5

    def normalized(self) -> Vector2:
        norm = self.norm()
        return Vector2(self.x / norm, self.y / norm)

    def distance(self, other: Vector2) -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def perp(self) -> Vector2:
        return Vector2(1, -self.x / self.y)

    def as_tuple(self) -> Tuple[float, float]:
        return self.x, self.y

    def as_int_tuple(self) -> Tuple[int, int]:
        return int(round(self.x)), int(round(self.y))

    def angle_with_x_axis(self, other: Vector2) -> float:
        diff = other - self
        rad = atan2(diff.y, diff.x)
        if rad < 0:
            rad += 2 * pi
        return degrees(rad)

    def __str__(self) -> str:
        return str(astuple(self))

    def __repr__(self) -> str:
        return f"Vector2 {asdict(self)}"


Vector2.ZERO = Vector2(0.0, 0.0)
Vector2.UP = Vector2(0.0, 1.0)
Vector2.DOWN = Vector2(0.0, -1.0)
Vector2.LEFT = Vector2(-1.0, 0.0)
Vector2.RIGHT = Vector2(1.0, 0.0)
