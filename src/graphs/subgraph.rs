use crate::{
    attr::Attr,
    edge::EdgeId,
    graphs::graph::{Graph, GraphId},
    node::NodeId,
    utils,
};

use std::borrow::Borrow;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::io::Write;

use rayon::prelude::*;

#[derive(Debug, Clone, Eq)]
/// A `SubGraph` holds indices of its own nodes and edges,
/// and its children subgraphs.
///
/// ```ignore
/// subgraph A {
///     subgraph B {
///         node C
///     }
/// }
/// ```
/// In such a case, `subgraph B` holds `node C`, not `subgraph A`.
pub struct SubGraph {
    /// Name of the subgraph
    pub(crate) id: GraphId,
    /// Ids of its children subgraphs, referenced in `Graph`
    pub(crate) subgraph_ids: HashSet<GraphId>,
    /// Ids of its own nodes, referenced in `Graph`
    pub(crate) node_ids: HashSet<NodeId>,
    /// Ids of its own edges, referenced in `Graph`
    pub(crate) edge_ids: HashSet<EdgeId>,
    /// Attributes of the graph in key, value mappings
    pub(crate) attrs: HashSet<Attr>,
}

impl PartialEq for SubGraph {
    fn eq(&self, other: &SubGraph) -> bool {
        self.id == other.id
    }
}

impl Hash for SubGraph {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl Borrow<GraphId> for SubGraph {
    fn borrow(&self) -> &GraphId {
        &self.id
    }
}

impl Borrow<str> for SubGraph {
    fn borrow(&self) -> &str {
        &self.id
    }
}

impl SubGraph {
    pub fn id(&self) -> &GraphId {
        &self.id
    }

    pub fn attrs(&self) -> &HashSet<Attr> {
        &self.attrs
    }

    pub fn subgraphs(&self) -> HashSet<&GraphId> {
        self.subgraph_ids.par_iter().map(|id| id).collect()
    }

    pub fn nodes(&self) -> HashSet<&NodeId> {
        self.node_ids.par_iter().map(|id| id).collect()
    }

    pub fn edges(&self) -> HashSet<&EdgeId> {
        self.edge_ids.par_iter().map(|id| id).collect()
    }

    pub(super) fn extract_nodes_and_edges(
        &self,
        node_ids: &HashSet<&NodeId>,
        edge_ids: &HashSet<&EdgeId>,
    ) -> SubGraph {
        let id = self.id.clone();

        let subgraph_ids = self.subgraph_ids.clone();

        let node_ids: HashSet<NodeId> =
            self.node_ids.par_iter().filter(|id| node_ids.contains(id)).cloned().collect();

        let edge_ids: HashSet<EdgeId> =
            self.edge_ids.par_iter().filter(|id| edge_ids.contains(id)).cloned().collect();

        let attrs = self.attrs.clone();

        SubGraph { id, subgraph_ids, node_ids, edge_ids, attrs }
    }

    pub(super) fn extract_subgraph(&self, subgraph_ids: &HashSet<&GraphId>) -> Option<SubGraph> {
        let subgraph_ids: HashSet<GraphId> =
            self.subgraph_ids.par_iter().filter(|id| subgraph_ids.contains(id)).cloned().collect();

        if subgraph_ids.is_empty() && self.node_ids.is_empty() && self.edge_ids.is_empty() {
            None
        } else {
            let id = self.id.clone();
            let node_ids = self.node_ids.clone();
            let edge_ids = self.edge_ids.clone();
            let attrs = self.attrs.clone();

            Some(SubGraph { id, subgraph_ids, node_ids, edge_ids, attrs })
        }
    }

    /// Write the graph to dot format.
    ///
    /// If true, `sort_nodes` will use a deterministic ordering (sorted by node
    /// name) for the nodes and subgraphs that are emitted. This can affect the
    /// layout Graphviz uses (it appears to consider node ordering in the input
    /// dot files for nodes within a rank when laying out nodes).
    pub(super) fn to_dot<W: ?Sized>(
        &self,
        graph: &Graph,
        indent: usize,
        sort_nodes: bool,
        writer: &mut W,
    ) -> std::io::Result<()>
    where
        W: Write,
    {
        let id = utils::pretty_id(&self.id);
        if indent == 0 {
            writeln!(writer, "digraph {id} {{")?;
        } else {
            (0..indent).try_for_each(|_| write!(writer, "\t"))?;
            writeln!(writer, "subgraph {id} {{")?;
        }

        if !self.attrs.is_empty() {
            (0..=indent).try_for_each(|_| write!(writer, "\t"))?;
            writeln!(writer, "graph [")?;

            for attr in &self.attrs {
                attr.to_dot(indent + 1, writer)?;
            }

            (0..=indent).try_for_each(|_| write!(writer, "\t"))?;
            writeln!(writer, "]")?;
        }

        let (mut sorted_subgraphs, mut unsorted_subgraphs);
        let subgraphs = if sort_nodes {
            let mut subgraphs = self.subgraph_ids.par_iter().collect::<Vec<_>>();
            subgraphs.par_sort();
            sorted_subgraphs = subgraphs.into_iter();
            &mut sorted_subgraphs as &mut dyn Iterator<Item = &_>
        } else {
            unsorted_subgraphs = self.subgraph_ids.iter();
            &mut unsorted_subgraphs as &mut dyn Iterator<Item = &_>
        };
        for id in subgraphs {
            let subgraph = graph.search_subgraph(id).unwrap();
            subgraph.to_dot(graph, indent + 1, sort_nodes, writer)?;
        }

        let (mut sorted_nodes, mut unsorted_nodes);
        let nodes = if sort_nodes {
            let mut nodes = self.node_ids.par_iter().collect::<Vec<_>>();
            nodes.par_sort();
            sorted_nodes = nodes.into_iter();
            &mut sorted_nodes
        } else {
            unsorted_nodes = self.node_ids.iter();
            &mut unsorted_nodes as &mut dyn Iterator<Item = &_>
        };
        for id in nodes {
            let node = graph.search_node(id).unwrap();
            node.to_dot(indent + 1, writer)?;
        }

        for id in &self.edge_ids {
            let edge = graph.search_edge(id).unwrap();
            edge.to_dot(indent + 1, writer)?;
        }

        (0..indent).try_for_each(|_| write!(writer, "\t"))?;

        writeln!(writer, "}}")?;

        Ok(())
    }
}
