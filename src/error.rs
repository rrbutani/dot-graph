use thiserror::Error;

use crate::prelude::EdgeId;

#[derive(Error, Debug)]
pub enum DotGraphError {
    #[error("`{0}` is not a valid dot graph")]
    InvalidGraph(String),
    #[error("`{0}` is not a digraph")]
    UndirectedGraph(String),
    #[error("`{0}` contains a cycle")]
    Cycle(String),
    #[error("`{0}` is not a node of graph `{1}`")]
    NoSuchNode(String, String),
    #[error("`{0}` is not an edge of graph `{1}`")]
    NoSuchEdge(String, String),
    #[error("`{0}` is not a subgraph of graph `{1}`")]
    NoSuchSubGraph(String, String),
    #[error("specified nodes do not belong to the same graph or subgraph; expected node `{node_id}` to be in `{expected_to_be_in_subgraph}` within graph `{graph_name}`")]
    NodesAreNotUnderSameParentSubgraph {
        node_id: String,
        expected_to_be_in_subgraph: String,
        graph_name: String,
    },
    #[error("a subgraph named `{subgraph_name}` already exists in graph `{graph_name}`")]
    SubgraphAlreadyExists { subgraph_name: String, graph_name: String },
    #[error("cannot construct a subgraph out of zero nodes")]
    CannotConstructEmptySubgraph,
    #[error("cannot remove node `{node}` of graph `{graph}`; it still has remaining edge(s) including an edge {dir} `{other_node}`", dir = if *forward { "to" } else { "from" })]
    NodeStillHasEdges {
        node: String,
        graph: String,
        forward: bool,
        other_node: String,
    },
    #[error("a node named `{node_name}` already exists in graph `{graph_name}`")]
    NodeAlreadyExists { node_name: String, graph_name: String },
    #[error("an edge with identity `{edge_id:?}` already exists in graph `{graph_name}`")]
    EdgeAlreadyExists { edge_id: Box<EdgeId>, graph_name: String },
    #[error(transparent)]
    IOError(#[from] std::io::Error),
}
