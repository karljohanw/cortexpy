import attr
from delegation import SingleDelegated

from cortexpy.graph import cortex
from cortexpy.graph.parser.random_access import load_ra_cortex_graph
from cortexpy.graph.parser.kmer import EmptyKmerBuilder
from cortexpy.test.builder import Graph


@attr.s(slots=True)
class CortexGraphBuilder(object):
    graph = attr.ib(attr.Factory(cortex.CortexDiGraph))
    colors = attr.ib(init=False)
    kmer_builder = attr.ib(attr.Factory(EmptyKmerBuilder))

    def __attrs_post_init__(self):
        self.with_colors(0)

    def with_node_coverage(self, node, coverage):
        if isinstance(coverage, int):
            coverage = (coverage,)
        else:
            coverage = tuple(coverage)
        assert len(self.colors) == len(coverage)
        self.graph.node[node].coverage = coverage
        return self

    def with_node_kmer(self, node, kmer):
        self.graph.add_node(node, kmer=kmer)
        return self

    def with_node(self, node):
        return self.add_node(node)

    def add_node(self, node):
        if node not in self.graph:
            self.with_node_kmer(node, self.kmer_builder.build_or_get(node))
        return self

    def with_colors(self, *colors):
        assert len(self.graph) == 0
        self.colors = set(colors)
        self.kmer_builder.num_colors = len(self.colors)
        return self

    def with_color(self, color):
        self.colors.add(color)
        self.kmer_builder.num_colors = len(self.colors)

    def add_edge(self, u, v, color=0, key=None):
        if key is not None:
            color = key
        self.add_edge_with_color(u, v, color)
        return self

    def add_edge_with_color(self, u, v, color):
        assert color in self.colors
        self.add_node(u)
        self.add_node(v)
        self.graph.add_edge(u, v, key=color)
        return self

    def add_path(self, *k_strings, color=0, coverage=0):
        if len(k_strings) == 1 and isinstance(k_strings[0], list):
            k_strings = k_strings[0]
        kmer = self.kmer_builder.build_or_get(k_strings[0])
        kmer.coverage = list(kmer.coverage)
        for cov_color in range(kmer.num_colors):
            kmer.coverage[cov_color] = coverage
        self.graph.add_node(k_strings[0], kmer=kmer)
        if len(k_strings) > 1:
            for k_string1, k_string2 in zip(k_strings[:-1], k_strings[1:]):
                kmer = self.kmer_builder.build_or_get(k_string2)
                kmer.coverage = list(kmer.coverage)
                for cov_color in range(kmer.num_colors):
                    kmer.coverage[cov_color] = coverage
                self.graph.add_node(k_string2, kmer=kmer)
                self.add_edge_with_color(k_string1, k_string2, color)

    def build(self):
        return self.graph


class CortexBuilder(SingleDelegated):

    def build(self):
        return load_ra_cortex_graph(self.delegate.build())


@attr.s()
class CortexGraphMappingBuilder(SingleDelegated):
    delegate = attr.ib()
    ra_parser_args = attr.ib(attr.Factory(dict))

    def build(self):
        return load_ra_cortex_graph(self.delegate.build(),
                                    ra_parser_args=self.ra_parser_args)._kmer_mapping

    def with_kmer_cache_size(self, n):
        self.ra_parser_args['kmer_cache_size'] = n
        return self


def get_cortex_builder():
    return CortexBuilder(Graph())


def get_cortex_graph_mapping_builder():
    return CortexGraphMappingBuilder(Graph())