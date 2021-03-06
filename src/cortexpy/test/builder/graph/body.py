import struct
from io import BytesIO
from math import ceil
from struct import pack

import attr
import numpy as np
from attr import Factory
from bitstring import BitArray

from cortexpy.edge_set import EdgeSet

UINT64_T = 8
KMER_LETTER_TO_NUM = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
KMER_LETTER_NUM_TO_BIT_REPR = ('00', '01', '10', '11')
KMER_CONTAINER_WORD_SIZE_IN_BITS = 64
LETTERS_PER_UINT64_T = UINT64_T * 4


def pack_edge_set(edge_set):
    edge_set = [str(flag) for flag in edge_set]
    edge = BitArray('0b' + ''.join(edge_set))
    return edge.tobytes()


@attr.s(slots=True)
class KmerRecord(object):
    kmer = attr.ib()
    coverage = attr.ib()
    edges = attr.ib()

    @property
    def kmer_container_size_in_uint64ts(self):
        return ceil(len(self.kmer) / LETTERS_PER_UINT64_T)

    def __attrs_post_init__(self):
        assert len(self.edges) == len(self.coverage)

    def to_bytestring(self):
        byte_string = self.pack_kmer()
        for color_coverage in self.coverage:
            byte_string += pack('I', color_coverage)
        for edge_set in self.edges:
            byte_string += pack_edge_set(edge_set)
        return byte_string

    def pack_kmer(self):
        kmer_string = ''.join(self.kmer).upper()
        kmer_longs = []
        kmer = 0
        for letter_idx, letter in enumerate(reversed(kmer_string)):
            if letter_idx % LETTERS_PER_UINT64_T == 0 and letter_idx > 0:
                kmer_longs.append(kmer)
                kmer = 0
            kmer |= KMER_LETTER_TO_NUM[letter] << (letter_idx % LETTERS_PER_UINT64_T) * 2
        kmer_longs.append(kmer)
        kmer_longs = list(reversed(kmer_longs))
        kmer_bytes = b''
        for long in kmer_longs:
            kmer_bytes += struct.pack('<Q', long)
        return kmer_bytes


@attr.s(slots=True)
class Body(object):
    sort_kmers = attr.ib(True)
    kmer_records = attr.ib(Factory(list), init=False)
    kmer_size = attr.ib(None)
    _kmer_container_size = attr.ib(None)

    @property
    def kmer_container_size(self):
        if self._kmer_container_size is None:
            if self.kmer_size is None:
                raise Exception("Need to call with_kmer_record to set kmer size first")
            self._kmer_container_size = max(1, ceil(self.kmer_size / 32))
        return self._kmer_container_size

    @kmer_container_size.setter
    def kmer_container_size(self, value):
        self._kmer_container_size = value

    def with_kmer_record(self, kmer):
        if self.kmer_size is None:
            self.kmer_size = len(kmer.kmer)
        assert len(kmer.kmer) == self.kmer_size

        coverage_per_color, edges_per_color = kmer.coverage, kmer.edges
        if isinstance(coverage_per_color, int):
            coverage_per_color = [coverage_per_color]
        if isinstance(edges_per_color, str):
            edges_per_color = [edges_per_color]
        assert self.kmer_container_size >= len(kmer.kmer) / 32
        self.kmer_records.append(
            KmerRecord(kmer.kmer, coverage_per_color, edges_per_color))
        return self

    def build(self):
        byte_strings = []
        if self.sort_kmers:
            self.kmer_records = sorted(self.kmer_records, key=lambda r: r.kmer)
        for record in self.kmer_records:
            byte_strings.append(record.to_bytestring())
        return BytesIO(b''.join(byte_strings))


def as_edge_set(edge_set_string):
    return EdgeSet(edge_set_string_to_array(edge_set_string))


def edge_set_string_to_array(edge_set_string):
    assert len(edge_set_string) == 8
    edge_set = []
    for edge in edge_set_string:
        if edge == '.':
            edge_set.append(0)
        else:
            edge_set.append(1)
    return np.array(edge_set[:4] + list(reversed(edge_set[4:])), dtype=np.uint8)
