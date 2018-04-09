def traverse(argv):
    import argparse
    from cortexpy.graph import traversal
    from .shared import get_shared_argsparse
    shared_parser = get_shared_argsparse()
    parser = argparse.ArgumentParser(
        'cortexpy traverse', parents=[shared_parser],
        description="""
        Traverse a cortex graph starting from each k-mer in an initial_contig and return the
        subgraph as a Python pickle object."""
    )
    parser.add_argument('--graphs', nargs='+',
                        required=True,
                        help="Input cortexpy graphs."
                             "  Multiple graphs can be specified and are joined on-the-fly.")
    parser.add_argument('initial_contig', help="Initial contig from which to start traversal")
    parser.add_argument('--orientation',
                        type=traversal.constants.EngineTraversalOrientation,
                        choices=[o.name for o in traversal.constants.EngineTraversalOrientation],
                        default=traversal.constants.EngineTraversalOrientation.both,
                        help='Traversal orientation')
    parser.add_argument('-c', '--colors',
                        nargs='+',
                        type=int,
                        help="""Colors to traverse.  May take multiple color numbers separated by
                        a space.  The traverser will follow all colors
                        specified.  Will follow all colors if not specified.
                        """, default=None)
    parser.add_argument('--initial-fasta', action='store_true',
                        help='Treat initial_contig as a file in FASTA format')
    parser.add_argument('--subgraphs', action='store_true',
                        help='Emit traversal as sequence of networkx subgraphs')
    parser.add_argument('--max-nodes', type=int, default=None,
                        help='Maximum number of nodes to traverse (int).'
                             '  Die without output if max nodes is exceeded')
    parser.add_argument('--logging-interval', type=int, default=90,
                        help='Logging interval.  [default: %(default)s]')
    parser.add_argument('--cache-size', type=int, default=0, help='Number of kmers to cache')
    parser.add_argument('--binary-search-cache-size', type=int, default=0,
                        help='Number of kmers to cache for binary search')
    args = parser.parse_args(argv)

    from cortexpy.logging_config import configure_logging_from_args
    configure_logging_from_args(args)

    import logging
    logger = logging.getLogger('cortexpy.traverse')

    import sys
    from contextlib import ExitStack
    with ExitStack() as stack:
        if args.out == '-':
            output = sys.stdout.buffer
        else:
            output = stack.enter_context(open(args.out, 'wb'))

        import networkx as nx
        from cortexpy.graph import parser as g_parser, traversal
        if len(args.graphs) == 1:
            ra_parser = g_parser.RandomAccess(
                stack.enter_context(open(args.graphs[0], 'rb')),
                kmer_cache_size=args.cache_size,
                kmer_binary_search_cache_size=args.binary_search_cache_size,
            )
        else:
            ra_parser = g_parser.RandomAccessCollection(
                [g_parser.RandomAccess(stack.enter_context(open(graph_path, 'rb')),
                                       kmer_cache_size=args.cache_size,
                                       kmer_binary_search_cache_size=args.binary_search_cache_size)
                 for graph_path in args.graphs])
        engine = traversal.Engine(
            ra_parser,
            orientation=traversal.constants.EngineTraversalOrientation[args.orientation.name],
            max_nodes=args.max_nodes,
            logging_interval=args.logging_interval
        )

        if args.colors is not None:
            engine.traversal_colors = args.colors
        else:
            engine.traversal_colors = tuple(list(range(engine.ra_parser.num_colors)))
        logger.info('Traversing colors: ' + ','.join([str(c) for c in engine.traversal_colors]))

        if args.initial_fasta:
            engine.traverse_from_each_kmer_in_fasta(args.initial_contig)
        else:
            engine.traverse_from_each_kmer_in(args.initial_contig)

        output_graph = engine.graph
        if args.subgraphs and len(output_graph) > 0:
            for subgraph in sorted(nx.weakly_connected_component_subgraphs(output_graph),
                                   key=lambda g: len(g), reverse=True):
                nx.write_gpickle(subgraph, output)
        else:
            nx.write_gpickle(output_graph, output)
