try:
    import mermaid as md
    from mermaid.graph import Graph
except:
    md = None
    Graph = None


def plot_mermaid(mermaid_diagram: str):
    """Render the graph in jupyter notebook."""
    if Graph is None and md is None:
        raise ImportError(
            "`mermaid` client is not available. "
            "Install with `pip install msgflux[plot]`."
        )    
    graph = Graph(
        title="chartflow",
        script=mermaid_diagram,
    )
    return md.Mermaid(graph)
