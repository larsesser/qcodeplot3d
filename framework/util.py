import enum


class Kind(enum.Enum):
    """Which kind of stabilizer?"""
    x = "x"
    z = "z"

    @property
    def opposite(self) -> "Kind":
        if self == Kind.x:
            return Kind.z
        if self == Kind.z:
            return Kind.x
        raise NotImplementedError
