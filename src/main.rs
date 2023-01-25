extern crate dot_graph;

use dot_graph::parser::parse;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let graph = parse(&path);

    /*
    println!("{:?}", graph.subgraphs);

    for node in &graph.nodes {
        println!("node [ id: {}, parent: {} ]", node.id, node.parent);
    }

    for edge in &graph.edges {
        println!("edge [ {} -> {} ]", edge.from, edge.to);
    }
    */

    println!("{}", graph.filter("graph1_subgraph34").unwrap().to_dot());
}
