import abc
from dataclasses import dataclass

from framework.layer import Syndrome


@dataclass
class Decoder(abc.ABC):

    @abc.abstractmethod
    def decode(self, syndrome: Syndrome) -> list[int]:
        """Determine the qubits on which a correction shall be applied from a given syndrome."""


# Mapping of syndromes (detector events) to qubits of the correction.
LookupTable = dict[tuple[bool, ...], list[int]]


@dataclass
class LookupTableDecoder(Decoder):
    lookup_table: LookupTable

    def decode(self, syndrome: Syndrome) -> list[int]:
        """Determine the qubits on which a correction shall be applied from a given syndrome.

        Use a (non-scalable) lookup table for decoding.
        """
        return self.lookup_table.get(syndrome.value, [])
