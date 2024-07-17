import abc
from dataclasses import dataclass

from framework.syndrome import Syndrome
from framework.util import Kind


@dataclass
class Decoder(abc.ABC):
    kind: Kind

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
