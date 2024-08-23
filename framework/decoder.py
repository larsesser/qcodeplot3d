import abc
from dataclasses import dataclass, field
from typing import Literal, overload

from framework.syndrome import Syndrome
from framework.util import Kind


@dataclass
class Decoder(abc.ABC):
    kind: Kind

    @overload
    def decode(self, syndrome: Syndrome, return_all_corrections: Literal[False] = False) -> list[int]:
        ...

    @overload
    def decode(self, syndrome: Syndrome, return_all_corrections: Literal[True]) -> list[list[int]]:
        ...

    @abc.abstractmethod
    def decode(self, syndrome: Syndrome, return_all_corrections: bool = False) -> list[int] | list[list[int]]:
        """Determine the qubits on which a correction shall be applied from a given syndrome."""


# Mapping of syndromes (detector events) to qubits of the correction.
LookupTable = dict[tuple[bool, ...], list[int]]


@dataclass
class LookupTableDecoder(Decoder):
    lookup_table: LookupTable
    missed_syndromes: int = field(default=0, init=False)

    def decode(self, syndrome: Syndrome, return_all_corrections: bool = False) -> list[int] | list[list[int]]:
        """Determine the qubits on which a correction shall be applied from a given syndrome.

        Use a (non-scalable) lookup table for decoding.
        """
        if syndrome.value not in self.lookup_table:
            self.missed_syndromes += 1
        return self.lookup_table.get(syndrome.value, [])
