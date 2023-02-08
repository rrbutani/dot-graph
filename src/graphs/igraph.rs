use crate::{
    edge::Edge, 
    graphs::{
        graph::{SubGraphIndex, NodeIndex, EdgeIndex},
        subgraph::SubGraph,
    }, 
    node::Node
};
use bimap::BiMap;
use rayon::prelude::*;
use std::mem::ManuallyDrop;
use std::ptr;

#[derive(Debug, Clone)]
/// An `IGraph` is an intermediate representation, to be transformed to `SubGraph` after parsing.
/// It holds ids of its children subgraphs, nodes, and edges.
///
/// `SubGraph` is a more compact form of an `IGraph`, in the sense that it holds indices of
/// children subgraphs, nodes, and edges to be referenced in `Graph`.
pub struct IGraph {
    /// Name of the igraph
    pub id: String,

    /// Ids of its children subgraphs
    pub subgraphs: Vec<String>,
    /// Its own nodes
    pub nodes: Vec<Node>,
    /// Its own edges
    pub edges: Vec<Edge>,
}

impl IGraph {
    /// Convert `IGraph` to `SubGraph`
    pub fn encode(
        &self,
        slookup: &BiMap<String, SubGraphIndex>,
        nlookup: &BiMap<String, NodeIndex>,
        elookup: &BiMap<(String, String), EdgeIndex>,
    ) -> SubGraph {
        let id = self.id.clone();

        let subgraph_idxs: Vec<SubGraphIndex> = (self.subgraphs.par_iter())
            .map(|subgraph| slookup.get_by_left(subgraph).unwrap())
            .cloned()
            .collect();

        let node_idxs: Vec<NodeIndex> = (self.nodes.par_iter())
            .map(|node| nlookup.get_by_left(&node.id).unwrap())
            .cloned()
            .collect();

        let edge_idxs: Vec<EdgeIndex> = (self.edges.par_iter())
            // https://users.rust-lang.org/t/hashmap-with-tuple-keys/12711/9
            // workaround to get &(String, String) from (&String, &String) without cloning
            .map(|edge| unsafe {
                let key = (ptr::read(&edge.from), ptr::read(&edge.to));
                let key = ManuallyDrop::new(key);

                elookup.get_by_left(&key).unwrap()
            })
            .cloned()
            .collect();

        SubGraph { id, subgraph_idxs, node_idxs, edge_idxs }
    }
}
