from micromind.geometry.vector import Vector2

VECTOR_1 = Vector2(5, 2)
VECTOR_2 = Vector2(5, 2)
VECTOR_3 = Vector2(3, 4)


class TestVector2:
    def test_zero(self):
        zero = Vector2.ZERO
        assert zero.x == 0.0
        assert zero.y == 0.0

    def test_sub(self):
        vector_sub = VECTOR_1 - VECTOR_2
        assert vector_sub.x == 0.0
        assert vector_sub.y == 0.0

    def test_add(self):
        vector_add = VECTOR_1 + VECTOR_2
        assert vector_add.x == 10.0
        assert vector_add.y == 4.0

    def test_mul(self):
        vector_mul = VECTOR_1 * 0.5
        assert vector_mul.x == 2.5
        assert vector_mul.y == 1.0

    def test_dot(self):
        dot_product = VECTOR_1.dot(VECTOR_2)
        assert dot_product == 29.0

    def test_norm(self):
        norm = VECTOR_3.norm()
        assert norm == 5.0

    def test_normalized(self):
        vector_normalized = VECTOR_3.normalized()
        assert vector_normalized.x == 0.6
        assert vector_normalized.y == 0.8
        assert vector_normalized.norm() == 1.0

    def test_distance(self):
        distance_zero = VECTOR_1.distance(VECTOR_1)
        assert distance_zero == 0.0
        distance_origin = VECTOR_1.distance(Vector2.ZERO)
        assert distance_origin == VECTOR_1.norm()

    def test_as_tuple(self):
        vector_tuple = VECTOR_1.as_tuple()
        assert vector_tuple == (5, 2)
        vector_int_tuple = (VECTOR_1 + Vector2(0.2, -0.3)).as_int_tuple()
        assert vector_int_tuple == (5, 2)
        assert type(vector_int_tuple[0]) is int
        assert type(vector_int_tuple[1]) is int

    def test_angle_with_x_axis(self):
        angle_right = Vector2.ZERO.angle_with_x_axis(Vector2.RIGHT)
        assert angle_right == 0.0

        angle_up = Vector2.ZERO.angle_with_x_axis(Vector2.UP)
        assert angle_up == 90.0

        angle_left = Vector2.ZERO.angle_with_x_axis(Vector2.LEFT)
        assert angle_left == 180.0

        angle_down = Vector2.ZERO.angle_with_x_axis(Vector2.DOWN)
        assert angle_down == 270.0

    def test_str(self):
        vector_str = str(VECTOR_1)
        assert vector_str == "(5, 2)"

    def test_repr(self):
        vector_repr = repr(VECTOR_1)
        assert vector_repr == "Vector2 {'x': 5, 'y': 2}"
