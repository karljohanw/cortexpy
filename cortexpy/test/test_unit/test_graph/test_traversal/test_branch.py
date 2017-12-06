import attr
import pytest

import cortexpy.graph.parser
import cortexpy.test.builder as builder
import cortexpy.test.expectation as expectation


@attr.s(slots=True)
class BranchTestDriver(object):
    graph_builder = attr.ib(attr.Factory(builder.Graph))
    start_kmer_string = attr.ib(None)

    def with_kmer(self, *args):
        self.graph_builder.with_kmer(*args)
        return self

    def with_kmer_size(self, n):
        self.graph_builder.with_kmer_size(n)
        return self

    def with_start_kmer_string(self, start_kmer_string):
        self.start_kmer_string = start_kmer_string
        return self

    def run(self):
        assert self.start_kmer_string is not None
        random_access_parser = cortexpy.graph.parser.RandomAccess(self.graph_builder.build())
        graph = (cortexpy.graph.traversal.Branch(random_access_parser, {})
                 .traverse_from(self.start_kmer_string))
        return expectation.graph.KmerGraphExpectation(graph)


class Test(object):
    def test_raises_on_empty_graph_returns_empty_graph(self):
        # given
        driver = BranchTestDriver().with_kmer_size(3).with_start_kmer_string('AAA')

        # when
        with pytest.raises(KeyError):
            driver.run()

    def test_two_unconnected_kmers_returns_graph_with_one_kmer(self):
        # given
        driver = (BranchTestDriver()
                  .with_kmer_size(3)
                  .with_kmer('AAA')
                  .with_kmer('AAT')
                  .with_start_kmer_string('AAA'))

        # when
        expect = driver.run()

        # then
        (expect.has_nodes('AAA')
         .has_n_edges(0))

    def test_two_connected_kmers_returns_graph_with_two_kmers(self):
        # given
        driver = (BranchTestDriver()
                  .with_kmer_size(3)
                  .with_kmer('AAA', 0, '.......T')
                  .with_kmer('AAT', 0, 'a.......')
                  .with_start_kmer_string('AAA'))

        # when
        expect = driver.run()

        # then
        (expect.has_nodes('AAA', 'AAT')
         .has_n_edges(0))