use crate::{
    attr::Attr,
    edge::{Edge, EdgeId},
    error::DotGraphError,
    graphs::{igraph::IGraph, subgraph::SubGraph},
    node::{Node, NodeId},
};

use std::{
    borrow::Borrow,
    collections::{HashMap, HashSet, VecDeque},
    fmt::Debug,
    hash::Hash,
    mem,
};
use std::{io::Write, mem::MaybeUninit, sync::Mutex};

use rayon::prelude::*;

pub type GraphId = String;

type SubTree = HashMap<GraphId, HashSet<GraphId>>;
type EdgeMap = HashMap<NodeId, HashSet<NodeId>>;

#[derive(Debug, Clone)]
/// A `Graph` serves as a database of the entire dot graph.
/// It holds all subgraphs, nodes, and edges in the graph as respective sets.
/// `SubGraph`s hold ids of its children, nodes, and edges
/// such that it can be referenced in `Graph`'s `subgraphs`, `nodes`, and `edges`.
///
/// **All subgraphs, nodes, and edges in the graph MUST HAVE UNIQUE IDS.**
pub struct Graph {
    /// Name of the entire graph
    id: GraphId,

    /// All subgraphs in the graph (subgraph ids must be unique)
    subgraphs: HashSet<SubGraph>,

    /// All nodes in the graph (node ids must be unique)
    nodes: HashSet<Node>,

    /// All edges in the graph (edge ids must be unique)
    edges: HashSet<Edge>,

    /// Parent-children relationships of the subgraphs
    subtree: SubTree,

    /// Map constructed from edges, in forward direction
    fwdmap: EdgeMap,
    /// Map constructed from edges, in backward direction
    bwdmap: EdgeMap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalkDirections {
    To,
    From,
    Both,
}

impl WalkDirections {
    pub fn to(&self) -> bool {
        *self != WalkDirections::From
    }
    pub fn from(&self) -> bool {
        *self != WalkDirections::To
    }
}

impl Graph {
    /// Constructs a new `graph`
    pub(crate) fn new(
        id: GraphId,
        root: IGraph,
        nodes: HashSet<Node>,
        edges: HashSet<Edge>,
    ) -> Result<Graph, DotGraphError> {
        let subgraphs: HashSet<SubGraph> = root.encode();

        let (fwdmap, bwdmap) = make_edge_maps(&nodes, &edges);

        let subtree = make_subtree(&subgraphs);

        let graph = Graph { id, subgraphs, nodes, edges, subtree, fwdmap, bwdmap };

        Ok(graph)
    }

    pub fn id(&self) -> &GraphId {
        &self.id
    }

    pub fn subgraphs_len(&self) -> usize { self.subgraphs.len() }
    pub fn nodes_len(&self) -> usize { self.nodes.len() }
    pub fn edges_len(&self) -> usize { self.edges.len() }

    pub fn subgraphs(&self) -> &HashSet<SubGraph> { &self.subgraphs }
    pub fn nodes(&self) -> &HashSet<Node> { &self.nodes }
    pub fn edges(&self) -> &HashSet<Edge> { &self.edges }

    /// **Warning**: Expensive!
    pub fn subgraphs_by_id(&self) -> HashSet<&GraphId> {
        self.subgraphs.par_iter().map(|subgraph| &subgraph.id).collect()
    }

    /// **Warning**: Expensive!
    pub fn nodes_by_id(&self) -> HashSet<&NodeId> {
        self.nodes.par_iter().map(|node| &node.id).collect()
    }

    /// **Warning**: Expensive!
    pub fn edges_by_id(&self) -> HashSet<&EdgeId> {
        self.edges.par_iter().map(|edge| &edge.id).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.subgraphs.is_empty() && self.nodes.is_empty() && self.edges.is_empty()
    }

    pub fn is_acyclic(&self) -> bool {
        self.topsort().is_ok()
    }

    /// Topologically sort nodes in this `Graph`.
    ///
    /// # Returns
    ///
    /// `Err` if this graph has a cycle, otherwise
    /// `Ok` with a vector of topologically sorted node ids.
    pub fn topsort(&self) -> Result<Vec<&NodeId>, DotGraphError> {
        let mut indegrees: HashMap<&NodeId, usize> = HashMap::new();
        for (to, froms) in &self.bwdmap {
            indegrees.insert(to, froms.len());
        }

        let mut visited: HashSet<&NodeId> = HashSet::new();

        let mut queue = VecDeque::new();
        let mut zero_indegrees: Vec<&NodeId> = indegrees
            .par_iter()
            .filter_map(|(&id, &indegree)| (indegree == 0).then_some(id))
            .collect();
        zero_indegrees.sort_unstable();

        for node in zero_indegrees {
            queue.push_back(node);
            visited.insert(node);
        }

        let mut sorted = Vec::new();
        while let Some(id) = queue.pop_front() {
            sorted.push(id);
            if let Some(tos) = self.fwdmap.get(id) {
                let mut tos = Vec::from_iter(tos);
                tos.sort_unstable();

                for to in tos {
                    let indegree = indegrees.get_mut(to).unwrap();
                    *indegree -= 1;
                    if *indegree == 0 {
                        queue.push_back(to);
                        visited.insert(to);
                    }
                }
            }
        }

        if sorted.len() == self.nodes.len() {
            Ok(sorted)
        } else {
            Err(DotGraphError::Cycle(self.id.clone()))
        }
    }

    /// Constructs a new `Graph`, containing only the given node ids.
    pub fn filter<'n, NodeKey>(&self, node_ids: impl IntoIterator<Item = &'n NodeKey>) -> Graph
    where
        Node: Borrow<NodeKey>,
        NodeKey: 'n + Debug + Hash + Eq + ?Sized,
    {
        self.extract(node_ids)
    }

    /// Constructs a new `Graph` given a root node and an (optional) depth
    /// limit.
    ///
    /// The returned graph will have all the transitive children of `root` up to
    /// the depth limit specified.
    pub fn children<NodeKey>(
        &self,
        root: &NodeKey,
        depth: Option<usize>,
    ) -> Result<Graph, DotGraphError>
    where
        Node: Borrow<NodeKey>,
        NodeKey: Debug + Hash + Eq + ?Sized,
    {
        self.select_subset(root, depth, WalkDirections::To)
    }

    pub fn parents<NodeKey>(
        &self,
        base: &NodeKey,
        depth: Option<usize>,
    ) -> Result<Graph, DotGraphError>
    where
        Node: Borrow<NodeKey>,
        NodeKey: Debug + Hash + Eq + ?Sized,
    {
        self.select_subset(base, depth, WalkDirections::From)
    }

    /// Constructs a new `Graph`, given a center node and an (optional) depth
    /// limit.
    ///
    /// # Arguments
    ///
    /// * `center` - Id of the center node
    /// * `depth` - Depth limit of the desired neighborhood
    ///
    /// # Returns
    ///
    /// `Err` if there is no node named `center`, `Ok` with neighbors `Graph`
    /// otherwise.
    pub fn neighbors<NodeKey>(
        &self,
        center: &NodeKey,
        depth: Option<usize>,
    ) -> Result<Graph, DotGraphError>
    where
        Node: Borrow<NodeKey>,
        NodeKey: Debug + Hash + Eq + ?Sized,
    {
        self.select_subset(center, depth, WalkDirections::Both)
    }

    pub fn select_subset<NodeKey>(
        &self,
        starting: &NodeKey,
        depth: Option<usize>,
        directions: WalkDirections,
    ) -> Result<Graph, DotGraphError>
    where
        Node: Borrow<NodeKey>,
        NodeKey: Debug + Hash + Eq + ?Sized,
    {
        let empty = HashSet::new();

        if let Some(starting) = self.nodes.get(starting) {
            let mut visited = HashSet::new();
            let mut frontier: VecDeque<(&NodeId, usize)> = VecDeque::new();
            frontier.push_back((&starting.id, 0));

            while let Some((id, vicinity)) = frontier.pop_front() {
                if depth.map(|d| vicinity > d).unwrap_or(false) || !visited.insert(id) {
                    continue;
                }

                let tos = if directions.to() { self.fwdmap.get(id).unwrap() } else { &empty };
                let froms = if directions.from() { self.bwdmap.get(id).unwrap() } else { &empty };
                let nexts = tos.union(froms);

                frontier.extend(nexts.map(|next| (next, vicinity + 1)));
            }

            Ok(self.extract::<NodeId>(visited.into_iter()))
        } else {
            Err(DotGraphError::NoSuchNode(format!("{:?}", starting), self.id.clone()))
        }
    }

    /// Extracts a subgraph of the graph into a new `Graph`.
    ///
    /// # Arguments
    ///
    /// * `root` - Id of the new root subgraph
    ///
    /// # Returns
    ///
    /// `Err` if there is no subgraph named `root`,
    /// `Ok` with subgraph-ed `Graph` otherwise.
    pub fn subgraph<SubgraphKey>(&self, root: &SubgraphKey) -> Result<Graph, DotGraphError>
    where
        SubGraph: Borrow<SubgraphKey>,
        GraphId: Borrow<SubgraphKey>,
        SubgraphKey: Debug + Hash + Eq + ?Sized,
    {
        self.collect_nodes(root).map_or(
            Err(DotGraphError::NoSuchSubGraph(format!("{:?}", root), self.id.clone())),
            |node_ids| Ok(self.extract(node_ids.into_iter())),
        )
    } // pull out subgraph into graph

    /// Adds the specified nodes to a new subgraph within the current graph.
    ///
    /// # Arguments
    ///
    /// * `subgraph_name` - Id to use for the new subgraph
    ///    + this must not collide with any subgraph id present in the graph
    ///
    /// * `node_ids` - List of nodes to move to the new subgraph
    ///    + all of these nodes must share the same graph/subgraph parent
    ///      - TODO: can we relax this requirement?
    ///    + this list cannot be empty
    pub fn add_subgraph<'n, NodeKey>(
        &mut self,
        subgraph_name: &GraphId,
        node_ids: impl IntoParallelIterator<Item = &'n NodeKey>,
    ) -> Result<(), DotGraphError>
    where
        Node: Borrow<NodeKey>,
        NodeKey: 'n + Debug + Hash + Eq + ?Sized,
    {
        // Check that the specified subgraph name does not already exist:
        if self.subgraphs().contains(subgraph_name) {
            return Err(DotGraphError::SubgraphAlreadyExists {
                subgraph_name: subgraph_name.clone(),
                graph_name: self.id.clone(),
            });
        }

        // Check that all the nodes exist:
        let nodes: HashSet<_> = node_ids
            .into_par_iter()
            .map(|id| {
                self.search_node(id)
                    .map(|n| n.id.clone())
                    .ok_or(DotGraphError::NoSuchNode(format!("{id:?}"), self.id.clone()))
            })
            .collect::<Result<_, _>>()?;

        // Find the immediate parent for each node.
        //
        // These must all match for the operation to succeed so to save time we:
        //   - find the parent for the first node given
        //   - check that all the nodes are under that node
        let Some(first_node) = nodes.iter().next() else {
            return Err(DotGraphError::CannotConstructEmptySubgraph)
        };

        let parent_subgraph_of_first_node = 'find_parent: {
            for subgraph in &self.subgraphs {
                if subgraph.node_ids.contains(first_node) {
                    break 'find_parent subgraph;
                }
            }

            // The root graph is within the subgraph map so we should never get
            // here...
            unreachable!()
        };

        for node in &nodes {
            if !parent_subgraph_of_first_node.node_ids.contains(node) {
                return Err(DotGraphError::NodesAreNotUnderSameParentSubgraph {
                    node_id: node.clone(),
                    expected_to_be_in_subgraph: parent_subgraph_of_first_node.id.clone(),
                    graph_name: self.id().clone(),
                });
            }
        }

        // With that out of the way we can begin manipulating the graph.
        let mut existing_sub = {
            // Careful not to manipulate the existing subgraph before removing
            // it from the HashSet -- that'd change the hash!
            let existing = parent_subgraph_of_first_node.clone();
            assert!(self.subgraphs.remove(&existing));
            existing
        };

        // Our nodes will migrate from the existing containing subgraph to our
        // new subgraph:
        for node in &nodes {
            let removed = existing_sub.node_ids.remove(node);
            debug_assert!(removed);
        }

        // All edges within the containing subgraph between nodes of the new
        // subgraph will also be migrated:
        let mut edges = HashSet::new();
        for edge in &existing_sub.edge_ids {
            if nodes.contains(&edge.from) && nodes.contains(&edge.to) {
                edges.insert(edge.clone());
            }
        }
        existing_sub.edge_ids = &existing_sub.edge_ids - &edges;

        // Finally, assemble the new subgraph:
        let new_sub = SubGraph {
            id: subgraph_name.clone(),
            subgraph_ids: HashSet::new(),
            node_ids: nodes,
            edge_ids: edges,
            attrs: existing_sub.attrs.clone(), // inherit attrs from parent
        };

        // Add it as a child of the existing subgraph:
        existing_sub.subgraph_ids.insert(subgraph_name.clone());
        self.subtree.get_mut(&existing_sub.id).unwrap().insert(subgraph_name.clone());

        // Note that our new subgraph has no child subgraphs:
        self.subtree.insert(subgraph_name.clone(), HashSet::new());

        // And insert both the updated old subgraph and the new subgraph:
        self.subgraphs.insert(existing_sub);
        self.subgraphs.insert(new_sub);

        Ok(())
    }

    fn extract<'n, NodeKey>(&self, node_ids: impl IntoIterator<Item = &'n NodeKey>) -> Graph
    where
        Node: Borrow<NodeKey>,
        NodeKey: 'n + Debug + Hash + Eq + ?Sized,
    {
        let mut nodes = HashSet::new();
        for id in node_ids {
            if let Some(node) = self.search_node(id) {
                nodes.insert(node.clone());
            }
        }
        let node_ids: HashSet<&NodeId> = nodes.par_iter().map(|node| &node.id).collect();

        let mut edges = HashSet::new();
        for edge in &self.edges {
            let from = &edge.id.from;
            let to = &edge.id.to;

            if node_ids.get(from).is_some() && node_ids.get(to).is_some() {
                edges.insert(edge.clone());
            }
        }
        let edge_ids: HashSet<&EdgeId> = edges.par_iter().map(|edge| &edge.id).collect();

        let subgraphs: HashSet<SubGraph> = self
            .subgraphs
            .par_iter()
            .map(|subgraph| subgraph.extract_nodes_and_edges(&node_ids, &edge_ids))
            .collect();

        let empty_subgraph_ids = empty_subgraph_ids(&subgraphs);
        let subgraph_ids: HashSet<&GraphId> = self
            .subgraphs
            .par_iter()
            .filter_map(|subgraph| {
                (!empty_subgraph_ids.contains(&subgraph.id)).then_some(&subgraph.id)
            })
            .collect();

        let subgraphs: HashSet<SubGraph> = subgraphs
            .par_iter()
            .filter_map(|subgraph| subgraph.extract_subgraph(&subgraph_ids))
            .collect();

        let (fwdmap, bwdmap) = make_edge_maps(&nodes, &edges);

        let subtree = make_subtree(&subgraphs);

        Graph { id: self.id.clone(), subgraphs, nodes, edges, subtree, fwdmap, bwdmap }
    }

    /// Returns a map from `Node`s to the `GraphId` of the subgraph they were
    /// removed from.
    pub fn remove_nodes<'n, NodeKey>(
        &mut self,
        node_ids: impl Clone + IntoIterator<Item = &'n NodeKey>,
    ) -> Result<HashMap<Node, GraphId>, DotGraphError>
    where
        Node: Borrow<NodeKey>,
        NodeKey: 'n + Debug + Hash + Eq + ?Sized,
    {
        // Do all our checks up-front so that we do not leave the graph in an
        // inconsistent state.

        // Check that all the specified nodes actually exist and have no
        // remaining edges:
        let mut count = 0;
        for node_id in node_ids.clone() {
            let Some(Node { id: node_id, .. }) = self.nodes.get(node_id) else {
                return Err(DotGraphError::NoSuchNode(format!("{node_id:?}"), self.id.clone()))
            };

            let fwd_edges = self.fwdmap[node_id].iter().map(|dest| (true, dest));
            let bwd_edges = self.bwdmap[node_id].iter().map(|src| (false, src));

            if let Some((forward, other_node)) = fwd_edges.chain(bwd_edges).next() {
                return Err(DotGraphError::NodeStillHasEdges {
                    node: node_id.clone(),
                    graph: self.id.clone(),
                    forward,
                    other_node: other_node.clone(),
                });
            }

            count += 1;
        }

        // With that out of the way we can begin removing nodes.
        let mut removed: HashMap<Node, MaybeUninit<GraphId>> = HashMap::with_capacity(count);
        for node_id in node_ids {
            let Some(node) = self.nodes.take(node_id) else {
                continue; // assuming duplicate..
            };

            self.fwdmap.remove(&node.id);
            self.bwdmap.remove(&node.id);

            assert!(removed.insert(node, MaybeUninit::uninit()).is_none());
        }

        // As part of the removal process we want to remove nodes from the
        // subgraphs they are in. Unfortunately we have no fast way to go from
        // node to containing subgraph...
        //
        // Here we process removals on the subgraphs separate from the prior
        // loop in the hopes that that may at least provide some
        // temporal-locality related cache wins.
        let subgraph_ids = self.subgraphs.iter().map(|s| s.id.clone()).collect::<Vec<_>>();
        let subgraphs_lock = Mutex::new(&mut self.subgraphs);
        let node_to_subgraph_map: HashMap<_, _> = subgraph_ids
            .into_par_iter()
            .flat_map(|subgraph_id| {
                let mut subgraphs = subgraphs_lock.lock().unwrap();
                let mut subgraph = subgraphs.take(&subgraph_id).unwrap();
                drop(subgraphs);

                // Iterate over the smaller of the two sets, looking for the
                // intersection:
                let mut intersection;
                if subgraph.node_ids.len() < removed.len() {
                    intersection = Vec::with_capacity(subgraph.node_ids.len());

                    for node_id in &subgraph.node_ids {
                        if let Some((node, _)) = removed.get_key_value::<NodeId>(node_id) {
                            // unfortunate clone; we cannot prove to `rustc`
                            // that we are only modifying the values of
                            // `removed`, not the keys (which we would like to
                            // reference here)
                            intersection.push(node.id.clone());
                        }
                    }

                    for node_id in &intersection {
                        assert!(subgraph.node_ids.remove(node_id));
                    }
                } else {
                    intersection = Vec::with_capacity(removed.len());

                    for (node, _) in &removed {
                        if subgraph.node_ids.contains(&node.id) {
                            intersection.push(node.id.clone()); // unfortunate clone
                            assert!(subgraph.node_ids.remove(&node.id));
                        }
                    }
                }

                let subgraph_id = subgraph.id.clone();
                let mut subgraphs = subgraphs_lock.lock().unwrap();
                assert!(subgraphs.insert(subgraph));
                drop(subgraphs);

                // the `subgraph_id` clones are unfortunate but alas: we're moving
                // our `subgraph`s because we have to because we cannot modify
                // HashSet elements in place so this becomes necessary :/
                intersection.into_par_iter().map(move |key| (key, subgraph_id.clone()))
            })
            .collect();

        removed.par_iter_mut().for_each(|(node, subgraph_id)| {
            subgraph_id.write(node_to_subgraph_map[&node.id].clone()); // unfortunate clone
        });

        Ok(unsafe { mem::transmute(removed) })
    }

    // `subgraph_id` defaults to the top-level subgraph if not specified
    pub fn add_node(
        &mut self,
        node: Node,
        mut subgraph_id: Option<GraphId>,
    ) -> Result<(), DotGraphError> {
        if self.nodes.contains(&node) {
            return Err(DotGraphError::NodeAlreadyExists {
                node_name: node.id,
                graph_name: self.id.clone(),
            });
        }

        if let Some(mut subgraph) =
            self.subgraphs.take(subgraph_id.get_or_insert_with(|| self.id.clone()))
        {
            assert!(subgraph.node_ids.insert(node.id.clone()));
            assert!(self.subgraphs.insert(subgraph));
        } else {
            return Err(DotGraphError::NoSuchSubGraph(
                format!("{:?}", subgraph_id.unwrap()),
                self.id.clone(),
            ));
        }

        assert!(self.fwdmap.insert(node.id.clone(), HashSet::new()).is_none());
        assert!(self.bwdmap.insert(node.id.clone(), HashSet::new()).is_none());

        assert!(self.nodes.insert(node));
        Ok(())
    }

    /// Returns a map from `Edge`s to the `GraphId` of the subgraph they were
    /// removed from.
    pub fn remove_edges<'e, EdgeKey>(
        &mut self,
        edge_ids: impl Clone + IntoIterator<Item = &'e EdgeKey>, // TODO: potential for shenanigans if this does not yield the same elements the second time around..
    ) -> Result<HashMap<Edge, GraphId>, DotGraphError>
    where
        Edge: Borrow<EdgeKey>,
        EdgeKey: 'e + Debug + Hash + Eq + ?Sized,
    {
        // Do all our checks up-front.

        // Check that the specified edges exist:
        let mut count = 0;
        for edge_id in edge_ids.clone() {
            let Some(_) = self.edges.get(edge_id) else {
                return Err(DotGraphError::NoSuchEdge(format!("{edge_id:?}"), self.id.clone()))
            };

            count += 1;
        }

        // With that out of the way we can remove the edges.
        let mut removed: HashMap<Edge, GraphId> = HashMap::with_capacity(count);
        for edge_id in edge_ids {
            let Some(edge) = self.edges.take(edge_id) else {
                continue; // assuming duplicate in the list..
            };
            let Edge { id: EdgeId { from, to, .. }, .. } = &edge;

            assert!(self.fwdmap.get_mut(from).unwrap().remove(to));
            assert!(self.bwdmap.get_mut(to).unwrap().remove(from));

            // As in `remove_node`, finding the subgraph an edge belongs to is
            // the hard part.
            //
            // For edges it is _more_ costly to copy ids (up to 4 `String`s
            // instead of just one) so we try a different (simpler) approach
            // for now (no parallelization, no grouping for subgraphs).
            for subgraph in &self.subgraphs {
                if subgraph.edge_ids.contains(&edge.id) {
                    let subgraph_name = subgraph.id.clone();

                    let mut subgraph = self.subgraphs.take(&subgraph_name).unwrap();
                    assert!(subgraph.edge_ids.remove(&edge.id));
                    self.subgraphs.insert(subgraph);

                    removed.insert(edge, subgraph_name);
                    break;
                }
            }
        }

        Ok(removed)

        // TODO: do we need to prune subgraphs (i.e. search for empty again?)
        //   - I think it's okay not to; if we want to we can call `extract`
        //     with all the nodes in the graph (expensive)
    }

    // `subgraph_id` defaults to the top-level subgraph if not specified
    pub fn add_edge(
        &mut self,
        edge: Edge,
        mut subgraph_id: Option<GraphId>,
    ) -> Result<(), DotGraphError> {
        if self.edges.contains(&edge) {
            return Err(DotGraphError::EdgeAlreadyExists {
                edge_id: Box::new(edge.id.clone()),
                graph_name: self.id.clone(),
            });
        }

        if let Some(mut subgraph) =
            self.subgraphs.take(subgraph_id.get_or_insert_with(|| self.id.clone()))
        {
            assert!(subgraph.edge_ids.insert(edge.id.clone()));
            assert!(self.subgraphs.insert(subgraph));
        } else {
            return Err(DotGraphError::NoSuchSubGraph(
                format!("{:?}", subgraph_id.unwrap()),
                self.id.clone(),
            ));
        }

        // Add to forward/backward edge maps:
        let Edge { id: EdgeId { from, to, .. }, .. } = &edge;
        assert!(self.fwdmap.entry(from.clone()).or_default().insert(to.clone()));
        assert!(self.bwdmap.entry(to.clone()).or_default().insert(from.clone()));

        assert!(self.edges.insert(edge));
        Ok(())
    }

    /// Search for a subgraph by `id`
    pub fn search_subgraph<SubgraphKey>(&self, id: &SubgraphKey) -> Option<&SubGraph>
    where
        SubGraph: Borrow<SubgraphKey>,
        SubgraphKey: Hash + Eq + ?Sized,
    {
        self.subgraphs.get(id)
    }

    /// Modify a subgraph's attributes
    pub fn modify_subgaph_attrs<SubgraphKey, R>(
        &mut self,
        id: &SubgraphKey,
        func: impl FnOnce(&mut HashSet<Attr>) -> R,
    ) -> Option<R>
    where
        SubGraph: Borrow<SubgraphKey>,
        SubgraphKey: Hash + Eq + ?Sized,
    {
        let mut sub = self.subgraphs.take(id)?;
        let ret = func(&mut sub.attrs);
        self.subgraphs.insert(sub);

        Some(ret)
    }

    /// Search for a node by `id`
    pub fn search_node<NodeKey>(&self, id: &NodeKey) -> Option<&Node>
    where
        Node: Borrow<NodeKey>,
        NodeKey: Hash + Eq + ?Sized,
    {
        self.nodes.get(id)
    }

    /// Modify a node's attributes
    pub fn modify_node_attrs<NodeKey, R>(
        &mut self,
        id: &NodeKey,
        func: impl FnOnce(&mut HashSet<Attr>) -> R,
    ) -> Option<R>
    where
        Node: Borrow<NodeKey>,
        NodeKey: Hash + Eq + ?Sized,
    {
        let mut node = self.nodes.take(id)?;
        let ret = func(&mut node.attrs);
        self.nodes.insert(node);

        Some(ret)
    }

    /// Search for an edge by `id`
    pub fn search_edge<EdgeKey>(&self, id: &EdgeKey) -> Option<&Edge>
    where
        Edge: Borrow<EdgeKey>,
        EdgeKey: Hash + Eq + ?Sized,
    {
        self.edges.get(id)
    }

    /// Modify an edge's attributes
    pub fn modify_edge_attrs<R, EdgeKey>(
        &mut self,
        id: &EdgeKey,
        func: impl FnOnce(&mut HashSet<Attr>) -> R,
    ) -> Option<R>
    where
        Edge: Borrow<EdgeKey>,
        EdgeKey: Hash + Eq + ?Sized,
    {
        let mut edge = self.edges.take(id)?;
        let ret = func(&mut edge.attrs);
        self.edges.insert(edge);

        Some(ret)
    }

    /// Get all children subgraphs by `id`
    ///
    /// # Returns
    ///
    /// `Err` if there is no subgraph with `id`,
    /// `Ok` with collected subgraph ids, where all ids are unique.
    /// (conceptually a set)
    pub fn collect_subgraphs<SubgraphKey>(
        &self,
        id: &SubgraphKey,
    ) -> Result<Vec<&GraphId>, DotGraphError>
    where
        GraphId: Borrow<SubgraphKey>,
        SubgraphKey: Debug + Hash + Eq + ?Sized,
    {
        if let Some(children) = self.subtree.get(id) {
            let subgraphs: Vec<&GraphId> =
                children.par_iter().map(|id| &self.search_subgraph(id).unwrap().id).collect();
            Ok(subgraphs)
        } else {
            Err(DotGraphError::NoSuchSubGraph(format!("{id:?}"), self.id.clone()))
        }
    }

    /// Collect all nodes in a subgraph by `id`
    ///
    /// # Returns
    ///
    /// `Err` if there is no subgraph with `id`,
    /// `Ok` with collected node ids, where all ids are unique.
    /// (conceptually a set)
    pub fn collect_nodes<SubgraphKey>(
        &self,
        id: &SubgraphKey,
    ) -> Result<Vec<&NodeId>, DotGraphError>
    where
        GraphId: Borrow<SubgraphKey>,
        SubGraph: Borrow<SubgraphKey>,
        SubgraphKey: Debug + Hash + Eq + ?Sized,
    {
        if let Some(children) = self.subtree.get(id) {
            let mut nodes = Vec::new();

            for id in children {
                nodes.extend(self.collect_nodes::<GraphId>(id).unwrap());
            }

            for id in &self.search_subgraph(id).unwrap().node_ids {
                nodes.push(&self.search_node(id).unwrap().id);
            }

            Ok(nodes)
        } else {
            Err(DotGraphError::NoSuchSubGraph(format!("{id:?}"), self.id.clone()))
        }
    }

    /// Collect all edges in a subgraph by `id`
    ///
    /// # Returns
    ///
    /// `Err` if there is no subgraph with `id`,
    /// `Ok` with collected edge ids, where all ids are unique.
    /// (conceptually a set)
    pub fn collect_edges<SubgraphKey>(
        &self,
        id: &SubgraphKey,
    ) -> Result<Vec<&EdgeId>, DotGraphError>
    where
        GraphId: Borrow<SubgraphKey>,
        SubGraph: Borrow<SubgraphKey>,
        SubgraphKey: Debug + Hash + Eq + ?Sized,
    {
        if let Some(children) = self.subtree.get(id) {
            let mut edges = Vec::new();

            for id in children {
                edges.extend(self.collect_edges::<GraphId>(id).unwrap());
            }

            for id in &self.search_subgraph(id).unwrap().edge_ids {
                edges.push(&self.search_edge(id).unwrap().id);
            }

            Ok(edges)
        } else {
            Err(DotGraphError::NoSuchSubGraph(format!("{id:?}"), self.id.clone()))
        }
    }

    /// Retrieve all nodes that are the predecessors of the node with `id`.
    ///
    /// # Returns
    ///
    /// `Err` if there is no node with `id`,
    /// `Ok` with a set of ids of predecessor nodes.
    pub fn froms<NodeKey>(&self, id: &NodeKey) -> Result<HashSet<&NodeId>, DotGraphError>
    where
        NodeId: Borrow<NodeKey>,
        NodeKey: Debug + Hash + Eq + ?Sized,
    {
        self.bwdmap
            .get(id)
            .map_or(Err(DotGraphError::NoSuchNode(format!("{id:?}"), self.id.clone())), |froms| {
                Ok(froms.par_iter().collect())
            })
    }

    /// Retrieve all nodes that are the successors of the node with `id`.
    ///
    /// # Returns
    ///
    /// `Err` if there is no node with `id`,
    /// `Ok` with a set of ids of successor nodes.
    pub fn tos<NodeKey>(&self, id: &NodeKey) -> Result<HashSet<&NodeId>, DotGraphError>
    where
        NodeId: Borrow<NodeKey>,
        NodeKey: Debug + Hash + Eq + ?Sized,
    {
        self.fwdmap
            .get(id)
            .map_or(Err(DotGraphError::NoSuchNode(format!("{id:?}"), self.id.clone())), |tos| {
                Ok(tos.par_iter().collect())
            })
    }

    /// Write the graph to dot format.
    pub fn to_dot<W: ?Sized>(&self, writer: &mut W) -> std::io::Result<()>
    where
        W: Write,
    {
        let root = self.subgraphs.get(&self.id).unwrap();

        root.to_dot(self, 0, writer)
    }
}

fn make_edge_maps(nodes: &HashSet<Node>, edges: &HashSet<Edge>) -> (EdgeMap, EdgeMap) {
    let mut fwdmap = EdgeMap::new();
    let mut bwdmap = EdgeMap::new();

    for edge in edges {
        let from = &edge.id.from;
        let to = &edge.id.to;

        fwdmap.entry(from.clone()).or_default().insert(to.clone());
        bwdmap.entry(to.clone()).or_default().insert(from.clone());
    }

    for node in nodes {
        let id = &node.id;

        fwdmap.entry(id.clone()).or_default();
        bwdmap.entry(id.clone()).or_default();
    }

    (fwdmap, bwdmap)
}

fn make_subtree(subgraphs: &HashSet<SubGraph>) -> SubTree {
    let mut subtree = HashMap::new();

    for subgraph in subgraphs {
        let children: HashSet<GraphId> = subgraph.subgraph_ids.par_iter().cloned().collect();
        subtree.insert(subgraph.id.clone(), children);
    }

    subtree
}

fn empty_subgraph_ids(subgraphs: &HashSet<SubGraph>) -> HashSet<GraphId> {
    let mut empty_subgraph_ids: HashSet<GraphId> = HashSet::new();

    loop {
        let updated_empty_subgraph_ids: HashSet<GraphId> = subgraphs
            .par_iter()
            .filter_map(|subgraph| {
                let nonempty_subgraph_ids: HashSet<&GraphId> = subgraph
                    .subgraph_ids
                    .par_iter()
                    .filter_map(|id| (!empty_subgraph_ids.contains(id)).then_some(id))
                    .collect();

                let is_empty = nonempty_subgraph_ids.is_empty()
                    && subgraph.node_ids.is_empty()
                    && subgraph.edge_ids.is_empty();

                is_empty.then_some(subgraph.id.clone())
            })
            .collect();

        if updated_empty_subgraph_ids.len() == empty_subgraph_ids.len() {
            break;
        }

        empty_subgraph_ids = updated_empty_subgraph_ids;
    }

    empty_subgraph_ids
}
