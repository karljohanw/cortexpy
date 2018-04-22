import io
import random
from unittest import mock

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as s
import cortexpy.test.builder as builder
import cortexpy.graph.parser as parser
from cortexpy.graph.parser import RandomAccess
from cortexpy.graph.parser.header import from_stream
from cortexpy.graph.parser.random_access import KmerUintSequence
import cortexpy.graph.serializer.kmer as kmer_serializer
from cortexpy.test.builder.graph.body import KmerRecord, as_edge_set
from cortexpy.test.builder.graph.kmer import kmers
from cortexpy.utils import lexlo


class TestDunderGetitemDunder(object):
    @given(s.data(),
           s.integers(min_value=1, max_value=3),
           s.integers(min_value=1, max_value=3),
           s.integers(min_value=0, max_value=5))
    def test_record_retrieval(self, data, kmer_size, num_colors, n_kmers):
        # given
        graph_builder = (builder.Graph()
                         .with_kmer_size(kmer_size)
                         .with_num_colors(num_colors))

        expected_kmers = []
        seen_kmers = set()
        for _ in range(n_kmers):
            kmer = data.draw(kmers(kmer_size, num_colors))
            while kmer.kmer in seen_kmers:
                kmer = data.draw(kmers(kmer_size, num_colors))
            seen_kmers.add(kmer.kmer)
            graph_builder.with_kmer_record(kmer)
            expected_kmers.append(kmer)

        cg = parser.RandomAccess(graph_builder.build())

        # when
        for expected_kmer in expected_kmers:
            kmer = cg[expected_kmer.kmer]

            # then
            assert expected_kmer.kmer == kmer.kmer
            assert np.all(expected_kmer.coverage == kmer.coverage)
            assert expected_kmer.edges == kmer.edges

    @given(s.integers(min_value=0, max_value=16))
    def test_cache_complexity(self, num_kmers):
        # given
        kmer_size = 11
        expected_num_calls = 2 * num_kmers
        cache_size = 0
        num_colors = 1
        b = builder.Graph() \
            .with_kmer_size(kmer_size) \
            .with_num_colors(num_colors)
        seen_kmers = set()
        for _ in range(num_kmers):
            kmer_string = lexlo(''.join([random.choice('ACGT') for _ in range(kmer_size)]))
            while kmer_string in seen_kmers:
                kmer_string = lexlo(''.join([random.choice('ACGT') for _ in range(kmer_size)]))
            seen_kmers.add(kmer_string)
            b.with_kmer(kmer_string)
        fh = b.build()

        ra = RandomAccess(fh, kmer_cache_size=cache_size)
        with mock.patch.object(fh, 'read', wraps=fh.read) as mocked_seek:
            # when
            for seen_kmer in sorted(seen_kmers):
                ra[seen_kmer]

            # then
            assert expected_num_calls == mocked_seek.call_count

    def test_raises_on_missing_kmer(self):
        # given
        graph_builder = builder.Graph()
        graph_builder.with_kmer_size(3)

        cg = parser.RandomAccess(graph_builder.build())

        # when
        with pytest.raises(KeyError):
            cg['AAA']


class TestGetKmerForString(object):
    def test_gets_aaa_for_ttt_query(self):
        # given
        graph_builder = builder.Graph()
        graph_builder.with_kmer_size(3)
        graph_builder.with_num_colors(1)

        expected_kmer = KmerRecord('AAA', [1], [as_edge_set('........')])
        graph_builder.with_kmer_record(expected_kmer)

        cg = parser.RandomAccess(graph_builder.build())

        # when
        assert expected_kmer.kmer == cg.get_kmer_for_string('AAA').kmer
        assert expected_kmer.kmer == cg.get_kmer_for_string('TTT').kmer


class TestDunderIterDunder(object):
    @given(s.data(),
           s.integers(min_value=1, max_value=10),
           s.integers(min_value=1, max_value=10),
           s.integers(min_value=0, max_value=4),
           s.booleans())
    def test_parses_records(self, data, kmer_size, num_colors, n_kmers, test_serializer):
        # given
        graph_builder = (builder.Graph()
                         .without_sorted_kmers()
                         .with_kmer_size(kmer_size)
                         .with_num_colors(num_colors))

        expected_kmers = []
        for _ in range(n_kmers):
            kmer = data.draw(kmers(kmer_size, num_colors))
            graph_builder.with_kmer_record(kmer)
            expected_kmers.append(kmer)
        ra_parser = parser.RandomAccess(graph_builder.build())

        if test_serializer:
            for real_kmer in ra_parser:
                buffer = io.BytesIO()
                real_kmer.dump(buffer)
                assert real_kmer._kmer_data._data == buffer.getvalue()
            buffer = io.BytesIO()
            kmer_serializer \
                .Kmers(ra_parser,
                       kmer_size=kmer_size,
                       num_colors=num_colors) \
                .dump(buffer)
            buffer.seek(0)
            ra_parser = parser.RandomAccess(buffer)

        # when
        for kmer, expected_kmer in zip(ra_parser, expected_kmers):
            # then
            assert expected_kmer.kmer == kmer.kmer
            assert np.all(expected_kmer.coverage == kmer.coverage)
            assert expected_kmer.edges == kmer.edges

    def test_gets_aaa(self):
        # given
        graph_builder = (builder.Graph()
                         .with_kmer_size(3)
                         .with_num_colors(1))

        expected_kmer = KmerRecord('AAA', (1,), [as_edge_set('........')])
        graph_builder.with_kmer_record(expected_kmer)

        cg = parser.RandomAccess(graph_builder.build())

        # when
        for kmer in cg:
            assert expected_kmer.kmer == kmer.kmer
            assert np.all(expected_kmer.coverage == kmer.coverage)
            assert expected_kmer.edges == kmer.edges

    def test_works_with_no_kmers(self):
        # given
        graph_builder = (builder.Graph()
                         .with_kmer_size(3)
                         .with_num_colors(1))

        cg = parser.RandomAccess(graph_builder.build())

        # when/then
        assert list(cg) == []


class TestKmerUintSequence(object):
    @given(s.data(),
           s.integers(min_value=1, max_value=129),
           s.integers(min_value=1, max_value=5))
    def test_index(self, data, kmer_size, n_kmers):
        # given
        num_colors = 1
        graph_builder = (builder.Graph()
                         .with_kmer_size(kmer_size)
                         .with_num_colors(num_colors))

        expected_kmers = []
        seen_kmers = set()
        for _ in range(n_kmers):
            kmer = data.draw(kmers(kmer_size, num_colors))
            while kmer.kmer in seen_kmers:
                kmer = data.draw(kmers(kmer_size, num_colors))
            seen_kmers.add(kmer.kmer)
            graph_builder.with_kmer_record(kmer)
            expected_kmers.append(kmer)
        expected_kmers = sorted(expected_kmers)

        graph_stream = graph_builder.build()
        header_stream = graph_builder.header.build()
        header = from_stream(header_stream)

        # when
        sequence = KmerUintSequence(graph_handle=graph_stream,
                                    body_start=len(header_stream.getvalue()),
                                    header=header,
                                    n_records=len(expected_kmers),
                                    kmer_cache_size=0)
        # then
        for idx, expected_kmer in enumerate(expected_kmers):
            # then
            assert idx == sequence.index_kmer_string(expected_kmer.kmer)
