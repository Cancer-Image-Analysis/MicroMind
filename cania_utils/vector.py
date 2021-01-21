

class Vector(object):
    # from https://stackoverflow.com/questions/57065080/draw-perpendicular-line-of-fixed-length-at-a-point-of-another-line
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def norm(self):
        return self.dot(self)**0.5

    def normalized(self):
        norm = self.norm()
        return Vector(self.x / norm, self.y / norm)

    def distance(self, other):
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5

    def perp(self):
        return Vector(1, -self.x / self.y)

    def as_tuple(self):
        return (self.x, self.y)

    def as_int_tuple(self):
        return (int(round(self.x)), int(round(self.y)))

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

    def __str__(self):
        return f'({self.x}, {self.y})'
