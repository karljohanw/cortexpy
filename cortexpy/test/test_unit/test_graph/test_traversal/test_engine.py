import attr
import pytest
from hypothesis import given
from hypothesis import strategies as s

import cortexpy.graph
import cortexpy.graph.parser
from cortexpy.graph import traversal
from cortexpy.graph.traversal import EngineTraversalOrientation
from cortexpy.test.expectation import KmerGraphExpectation
import cortexpy.test.builder as builder
import cortexpy.test.expectation as expectation


@attr.s(slots=True)
class EngineTestDriver(object):
    graph_builder = attr.ib(attr.Factory(builder.Graph))
    start_kmer_string = attr.ib(None)
    start_string = attr.ib(None)
    max_nodes = attr.ib(1000)
    traversal_orientation = attr.ib(EngineTraversalOrientation.original)
    traverser = attr.ib(None)
    traversal_colors = attr.ib((0,))

    def with_kmer(self, *args):
        self.graph_builder.with_kmer(*args)
        return self

    def with_kmer_size(self, n):
        self.graph_builder.with_kmer_size(n)
        return self

    def with_num_colors(self, n):
        self.graph_builder.with_num_colors(n)
        return self

    def with_start_kmer_string(self, start_kmer_string):
        self.start_kmer_string = start_kmer_string
        return self

    def with_start_string(self, start_string):
        self.start_string = start_string
        return self

    def with_max_nodes(self, max_nodes):
        self.max_nodes = max_nodes
        return self

    def with_traversal_orientation(self, orientation):
        self.traversal_orientation = EngineTraversalOrientation[orientation]
        return self

    def with_traversal_colors(self, *colors):
        self.traversal_colors = colors
        return self

    def run(self):
        random_access_parser = cortexpy.graph.parser.RandomAccess(self.graph_builder.build())
        self.traverser = traversal.Engine(random_access_parser,
                                          traversal_colors=self.traversal_colors,
                                          max_nodes=self.max_nodes,
                                          orientation=self.traversal_orientation)
        assert (self.start_string is None) != (self.start_kmer_string is None)
        if self.start_string:
            self.traverser.traverse_from_each_kmer_in(self.start_string)
        else:
            self.traverser.traverse_from(self.start_kmer_string)
        return expectation.graph.KmerGraphExpectation(self.traverser.graph)


class Test(object):
    def test_raises_on_empty(self):
        # given
        driver = EngineTestDriver()
        (driver
            .with_kmer_size(3)
            .with_start_kmer_string('AAA'))

        # when/then
        with pytest.raises(KeyError):
            driver.run()

        assert len(driver.traverser.graph) == 0

    def test_three_connected_kmers_returns_graph_with_three_kmers(self):
        # given
        driver = EngineTestDriver()
        (driver
            .with_kmer_size(3)
            .with_kmer('AAA', 0, '.......T')
            .with_kmer('AAT', 0, 'a....C..')
            .with_kmer('ATC', 0, 'a.......')
            .with_start_kmer_string('AAA'))

        # when
        expect = driver.run()

        # then
        (expect
            .has_nodes('AAA', 'AAT', 'ATC')
            .has_edges('AAA AAT 0', 'AAT ATC 0'))

    @given(s.integers(min_value=0, max_value=1))
    def test_three_connected_kmers_returns_graph_with_three_kmers_for_two_colors(self,
                                                                                 traversal_color):
        # given
        driver = EngineTestDriver() \
            .with_kmer_size(3) \
            .with_num_colors(2) \
            .with_traversal_colors(traversal_color) \
            .with_start_kmer_string('AAA')
        driver \
            .with_kmer('AAA 1 1 .......T .......T') \
            .with_kmer('AAT 1 1 a....C.. a....C..') \
            .with_kmer('ATC 1 1 a....... a.......')

        # when
        expect = driver.run()

        # then
        for node in ['AAA', 'AAT', 'ATC']:
            expect.has_node(node).has_coverages(1, 1)
        expect.has_n_nodes(3)
        expect.has_edges('AAA AAT 0',
                         'AAA AAT 1',
                         'AAT ATC 0',
                         'AAT ATC 1')

    def test_four_connected_kmers_in_star_returns_graph_with_four_kmers(self):
        # given
        driver = EngineTestDriver()
        (driver
            .with_kmer_size(3)
            .with_kmer('AAA', 0, '.......T')
            .with_kmer('AAT', 0, 'a....CG.')
            .with_kmer('ATC', 0, 'a.......')
            .with_kmer('ATG', 0, 'a.......')
            .with_start_kmer_string('AAA'))

        # when
        expect = driver.run()

        # then
        (expect
            .has_nodes('AAA', 'AAT', 'ATC', 'ATG')
            .has_edges('AAA AAT 0', 'AAT ATC 0', 'AAT ATG 0'))

    def test_cycle_is_traversed_once(self):
        # given
        driver = EngineTestDriver()
        (driver
            .with_kmer_size(3)
            .with_kmer('CAA', 0, '....A...')
            .with_kmer('AAA', 0, '.c.t...T')
            .with_kmer('AAT', 0, 'a...A...')
            .with_kmer('ATA', 0, 'a...A...')
            .with_kmer('TAA', 0, 'a...A...')
            .with_start_kmer_string('CAA'))

        # when
        expect = driver.run()

        # then
        (expect
            .has_nodes('CAA', 'AAA', 'AAT', 'ATA', 'TAA')
            .has_n_edges(5))

    def test_cycle_and_branch_is_traversed_once(self):
        # given
        driver = EngineTestDriver()
        (driver
            .with_kmer_size(3)
            .with_kmer('CAA', 0, '....A...')
            .with_kmer('AAA', 0, '.c.t.C.T')
            .with_kmer('AAC', 0, 'a.......')
            .with_kmer('AAT', 0, 'a...A...')
            .with_kmer('ATA', 0, 'a...A...')
            .with_kmer('TAA', 0, 'a...A...')
            .with_start_kmer_string('CAA'))

        # when
        expect = driver.run()

        # then
        (expect
            .has_nodes('CAA', 'AAA', 'AAT', 'ATA', 'TAA', 'AAC')
            .has_n_edges(6))

    def test_two_cycles_are_traversed_once(self):
        # given
        driver = EngineTestDriver()
        (driver
            .with_kmer_size(3)
            .with_kmer('CAA', 0, 'a...A...')
            .with_kmer('AAA', 0, '.c.t.C.T')
            .with_kmer('AAC', 0, 'a...A...')
            .with_kmer('ACA', 0, 'a...A...')
            .with_kmer('AAT', 0, 'a...A...')
            .with_kmer('ATA', 0, 'a...A...')
            .with_kmer('TAA', 0, 'a...A...')
            .with_start_kmer_string('CAA'))

        # when
        expect = driver.run()

        # then
        (expect
            .has_nodes('CAA', 'AAA', 'AAT', 'ATA', 'TAA', 'AAC', 'ACA')
            .has_n_edges(8))

    def test_two_cycles_are_traversed_once_in_revcomp(self):
        # given
        driver = EngineTestDriver()
        (driver
            .with_kmer_size(3)
            .with_kmer('CAA', 0, 'a...A...')
            .with_kmer('AAA', 0, '.c.t.C.T')
            .with_kmer('AAC', 0, 'a...A...')
            .with_kmer('ACA', 0, 'a...A...')
            .with_kmer('AAT', 0, 'a...A...')
            .with_kmer('ATA', 0, 'a...A...')
            .with_kmer('TAA', 0, 'a...A...')
            .with_start_kmer_string('TTG'))

        # when
        expect = driver.run()

        # then
        (expect
            .has_nodes('TTG', 'TTT', 'ATT', 'TAT', 'TTA', 'GTT', 'TGT')
            .has_n_edges(8))

    def test_exploration_of_two_colors_returns_all_kmers(self):
        # given
        driver = EngineTestDriver() \
            .with_kmer_size(3) \
            .with_num_colors(2) \
            .with_traversal_colors(0, 1) \
            .with_start_kmer_string('AAA')
        driver \
            .with_kmer('AAA 0 1 ........ .......T') \
            .with_kmer('AAT 1 1 .....C.. a.......') \
            .with_kmer('ATC 1 0 a....... ........')

        # when
        expect = driver.run()

        # then
        expect.has_node_coverages('AAA 0 1',
                                  'AAT 1 1',
                                  'ATC 1 0')
        expect.has_edges('AAA AAT 1',
                         'AAT ATC 0')

    def test_exploration_of_two_colors_on_initial_branch_point_returns_all_kmers(self):
        # given
        driver = EngineTestDriver() \
            .with_kmer_size(3) \
            .with_num_colors(2) \
            .with_traversal_colors(0, 1) \
            .with_start_kmer_string('AAA')
        driver \
            .with_kmer('AAA 1 1 .......T .....C..') \
            .with_kmer('AAT 1 0 a....... ........') \
            .with_kmer('AAC 0 1 ........ a.......')

        # when
        expect = driver.run()

        # then
        expect.has_node_coverages('AAA 1 1',
                                  'AAT 1 0',
                                  'AAC 0 1')
        expect.has_edges('AAA AAT 0',
                         'AAA AAC 1', )


class TestTraversalOrientationBoth(object):
    def test_with_three_linked_kmers_returns_graph_of_three_kmers(self):
        # given
        driver = EngineTestDriver()
        (driver
            .with_kmer_size(3)
            .with_kmer('AAA', 0, '.......T')
            .with_kmer('AAT', 0, 'a....C..')
            .with_kmer('ATC', 0, 'a.......')
            .with_start_kmer_string('ATC')
            .with_traversal_orientation('both'))

        # when
        expect = driver.run()

        # then
        (expect
            .has_nodes('AAA', 'AAT', 'ATC')
            .has_n_edges(2))


class TestReverseOrientation(object):
    def test_three_connected_kmers_returns_graph_with_three_kmers(self):
        # given
        driver = EngineTestDriver()
        (driver
            .with_kmer_size(3)
            .with_kmer('AAA', 0, '.......T')
            .with_kmer('AAT', 0, 'a....C..')
            .with_kmer('ATC', 0, 'a.......')
            .with_start_kmer_string('ATC')
            .with_traversal_orientation('reverse'))

        # when
        expect = driver.run()

        # then
        (expect
            .has_nodes('AAA', 'AAT', 'ATC')
            .has_edges(('AAA', 'AAT', 0), ('AAT', 'ATC', 0)))

    @given(s.sampled_from(('AAA', 'AAT', 'ATA', 'TAA')))
    def test_cycle_is_traversed_once(self, start_kmer_string):
        # given
        driver = EngineTestDriver()
        (driver
            .with_kmer_size(3)
            .with_kmer('CAA', 0, '....A...')
            .with_kmer('AAA', 0, '.c.t...T')
            .with_kmer('AAT', 0, 'a...A...')
            .with_kmer('ATA', 0, 'a...A...')
            .with_kmer('TAA', 0, 'a...A...')
            .with_traversal_orientation('reverse')
            .with_start_kmer_string(start_kmer_string))

        # when
        expect = driver.run()

        # then
        (expect
            .has_nodes('CAA', 'AAA', 'AAT', 'ATA', 'TAA')
            .has_n_edges(5))


class TestBothOrientation(object):
    @given(s.sampled_from(('CCA', 'CAA', 'AAA', 'AAT', 'ATA', 'TAA')))
    def test_cycle_and_branch_are_traversed_once(self, start_kmer_string):
        # given
        driver = EngineTestDriver()
        (driver
            .with_kmer_size(3)
            .with_kmer('CCA', 0, '....A...')
            .with_kmer('CAA', 0, '.c..A...')
            .with_kmer('AAA', 0, '.c.t...T')
            .with_kmer('AAT', 0, 'a...A...')
            .with_kmer('ATA', 0, 'a...A...')
            .with_kmer('TAA', 0, 'a...A...')
            .with_traversal_orientation('both')
            .with_start_kmer_string(start_kmer_string))

        # when
        expect = driver.run()

        # then
        (expect
            .has_nodes('CCA', 'CAA', 'AAA', 'AAT', 'ATA', 'TAA')
            .has_edges(('CCA', 'CAA', 0),
                       ('CAA', 'AAA', 0),
                       ('AAA', 'AAT', 0),
                       ('AAT', 'ATA', 0),
                       ('ATA', 'TAA', 0),
                       ('TAA', 'AAA', 0)))

    @given(s.data())
    def test_star_with_two_colors(self, data):
        kmers = ('CAA', 'GAA', 'AAA', 'AAT', 'AAC')
        start_kmer_string = data.draw(s.sampled_from(kmers))

        # given
        driver = EngineTestDriver()
        (driver
            .with_kmer_size(3)
            .with_num_colors(2)
            .with_kmer('CAA 1 1 ....A... ....A...')
            .with_kmer('GAA 1 1 ....A... ....A...')
            .with_kmer('AAA 1 1 .cg..C.T .cg.....')
            .with_kmer('AAT 1 0 a....... ........')
            .with_kmer('AAC 1 0 a....... ........')
            .with_traversal_orientation('both')
            .with_start_kmer_string(start_kmer_string))

        # when
        expect = driver.run()

        # then
        for node in ['CAA', 'GAA', 'AAA']:
            expect.has_node(node).has_coverages(1, 1)
        for node in ['AAT', 'AAC']:
            expect.has_node(node).has_coverages(1, 0)
        (expect
            .has_n_nodes(len(kmers))
            .has_edges('CAA AAA 0',
                       'CAA AAA 1',
                       'GAA AAA 0',
                       'GAA AAA 1',
                       'AAA AAT 0',
                       'AAA AAC 0'))


class TestTraverseFromEachKmerIn(object):
    def test_does_not_raise_on_empty(self):
        # given
        driver = EngineTestDriver()
        (driver
            .with_kmer_size(3)
            .with_start_string('AAA'))

        # when/then
        driver.run()

        assert len(driver.traverser.graph) == 0

    def test_cycle_and_branch_are_traversed_once(self):
        # given
        start_string = 'CCAAATAA'
        driver = EngineTestDriver()
        (driver
            .with_kmer_size(3)
            .with_kmer('CCA', 0, '....A...')
            .with_kmer('CAA', 0, '.c..A...')
            .with_kmer('AAA', 0, '.c.t...T')
            .with_kmer('AAT', 0, 'a...A...')
            .with_kmer('ATA', 0, 'a...A...')
            .with_kmer('TAA', 0, 'a...A...')
            .with_traversal_orientation('both')
            .with_start_string(start_string))

        # when
        expect = driver.run()

        # then
        (expect
            .has_nodes('CCA', 'CAA', 'AAA', 'AAT', 'ATA', 'TAA')
            .has_edges(('CCA', 'CAA', 0),
                       ('CAA', 'AAA', 0),
                       ('AAA', 'AAT', 0),
                       ('AAT', 'ATA', 0),
                       ('ATA', 'TAA', 0),
                       ('TAA', 'AAA', 0)))


class TestMaxNodes(object):
    def test_of_two_returns_with_two_nodes_plus_edges(self):
        # given
        driver = EngineTestDriver()
        (driver
            .with_kmer_size(3)
            .with_max_nodes(2)
            .with_kmer('AAA', 0, '.......T')
            .with_kmer('AAT', 0, 'a....CG.')
            .with_kmer('ATC', 0, 'a......T')
            .with_kmer('ATG', 0, 'a.......')
            .with_kmer('AGA', 0, '.......T')
            .with_start_kmer_string('AAA'))

        # when
        expect = driver.run()

        # then
        (expect
            .has_nodes('AAA', 'AAT', 'ATC', 'ATG')
            .has_n_edges(3))


class TestStartStringSize(object):
    def test_raises_when_string_wrong_size(self):
        for start_string in ['AAA', 'AAAAAAA']:
            # given
            driver = EngineTestDriver().with_kmer_size(5).with_start_kmer_string(start_string)

            with pytest.raises(AssertionError):
                driver.run()


class TestEdgeAnnotation(object):
    def test_with_single_kmer_and_link_annotates_etra_links(self):
        driver = EngineTestDriver() \
            .with_kmer_size(3) \
            .with_kmer('AAA 1 1 ........ .c...C..') \
            .with_start_kmer_string('AAA') \
            .with_traversal_colors(0)

        # when
        graph_expectation = driver.run()
        annotated_graph = cortexpy.graph.traversal.engine \
            .annotate_kmer_graph_edges(graph_expectation.graph)
        expect = KmerGraphExpectation(annotated_graph)

        # then
        expect.has_node('CAA').has_coverages(0, 0)
        expect.has_node('AAA').has_coverages(1, 1)
        expect.has_node('AAC').has_coverages(0, 0)
        expect.has_n_nodes(3)
        expect.has_edges('AAA AAC 1', 'CAA AAA 1')

    def test_for_revcomp_with_single_kmer_and_link_annotates_etra_links(self):
        driver = EngineTestDriver() \
            .with_kmer_size(3) \
            .with_kmer('AAA 1 1 ........ ..g..C..') \
            .with_traversal_colors(0) \
            .with_start_kmer_string('TTT')

        # when
        graph_expectation = driver.run()
        annotated_graph = cortexpy.graph.traversal.engine \
            .annotate_kmer_graph_edges(graph_expectation.graph)
        expect = KmerGraphExpectation(annotated_graph)

        # then
        expect.has_node('TTC').has_coverages(0, 0)
        expect.has_node('TTT').has_coverages(1, 1)
        expect.has_node('GTT').has_coverages(0, 0)
        expect.has_n_nodes(3)
        expect.has_edges('TTT TTC 1', 'GTT TTT 1')
