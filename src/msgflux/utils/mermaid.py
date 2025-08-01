import mermaid as md
from mermaid.graph import Graph


def plot_mermaid(mermaid_diagram):
    """Render the graph in jupyter notebook."""
    graph = Graph(
        title="chartflow",
        script=mermaid_diagram,
    )
    return md.Mermaid(graph)
