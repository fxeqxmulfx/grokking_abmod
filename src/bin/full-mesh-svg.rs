//! Draw every connection of the Rust full-mesh candidates as standalone SVGs.

use std::{fmt::Write as _, fs, path::PathBuf};

use grokking_abmod::{
    Result,
    simple::{Architecture, Connectivity, Encoding, SimpleConfig, TwoParentLayer},
};

fn draw(config: SimpleConfig) -> Result<String> {
    config.validate()?;
    let layers = config.two_parent_layers();
    let spacing = if config.depth > 16 { 34.0 } else { 150.0 };
    let (left, top, height) = (95.0, 145.0, 1024.0);
    let width = left + spacing * layers.len() as f64 + 100.0;
    let canvas_height = top + height + 45.0;
    let x = |column: usize| left + spacing * column as f64;
    let y = |node: usize, count: usize| top + (node as f64 + 0.5) * height / count as f64;
    let radius = if config.depth > 16 { 1.65 } else { 2.6 };
    let neurons: usize = layers.iter().map(|layer| layer.outputs).sum();
    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{canvas_height}\" viewBox=\"0 0 {width} {canvas_height}\" role=\"img\" aria-labelledby=\"title desc\">\n\
         <title id=\"title\">Full {} mesh: every neuron and connection</title>\n\
         <desc id=\"desc\">14 ordinary operand bits, {} hidden layers of 128 neurons, and 97 output neurons. Each neuron has exactly two incoming activation connections and its own bias. No connections are omitted. Zoom to inspect nodes and their parent tooltips.</desc>\n\
         <rect width=\"100%\" height=\"100%\" fill=\"#0a111c\"/>\n\
         <g font-family=\"system-ui, sans-serif\" fill=\"#edf3fb\">\n\
         <text x=\"30\" y=\"37\" font-size=\"25\">Full {} mesh</text>\n\
         <text x=\"30\" y=\"66\" font-size=\"14\" fill=\"#a9b6c8\">14 input bits · {} hidden layers × 128 neurons · 97 outputs</text>\n\
         <text x=\"30\" y=\"89\" font-size=\"13\" fill=\"#a9b6c8\">{} connections · {} biases · {} parameters · all connections shown</text>\n\
         <path d=\"M30 111h24\" stroke=\"#55c8f1\" stroke-width=\"2\"/><text x=\"62\" y=\"115\" font-size=\"12\">First parent</text>\n\
         <path d=\"M178 111h24\" stroke=\"#f3ad6a\" stroke-width=\"2\"/><text x=\"210\" y=\"115\" font-size=\"12\">Second parent</text>\n\
         <text x=\"370\" y=\"115\" font-size=\"12\" fill=\"#a9b6c8\">Connections point from left to right. Zoom to inspect.</text>\n</g>\n",
        config.connectivity,
        config.depth,
        config.connectivity,
        config.depth,
        2 * neurons,
        neurons,
        config.parameter_count()
    );
    for (index, layer) in layers.iter().enumerate() {
        writeln!(
            svg,
            "<g id=\"connections-{}\" fill=\"none\" stroke-width=\"0.65\" stroke-opacity=\"0.37\">",
            index + 1
        )?;
        for output in 0..layer.outputs {
            for (kind, parent) in layer.parents(output).into_iter().enumerate() {
                let color = if kind == 0 { "#55c8f1" } else { "#f3ad6a" };
                writeln!(
                    svg,
                    "<path data-layer=\"{}\" data-parent=\"{parent}\" data-node=\"{output}\" d=\"M{:.2} {:.2}L{:.2} {:.2}\" stroke=\"{color}\"><title>Layer {}, parent {} to neuron {output}</title></path>",
                    index + 1,
                    x(index),
                    y(parent, layer.inputs),
                    x(index + 1),
                    y(output, layer.outputs),
                    index + 1,
                    parent
                )?;
            }
        }
        writeln!(svg, "</g>")?;
    }
    writeln!(
        svg,
        "<g font-family=\"system-ui, sans-serif\" fill=\"#a9b6c8\" font-size=\"10\" text-anchor=\"middle\">"
    )?;
    writeln!(svg, "<text x=\"{}\" y=\"135\">Input</text>", x(0))?;
    for column in 1..=config.depth {
        writeln!(svg, "<text x=\"{}\" y=\"135\">{column}</text>", x(column))?;
    }
    writeln!(
        svg,
        "<text x=\"{}\" y=\"135\">Output</text></g>",
        x(layers.len())
    )?;
    writeln!(
        svg,
        "<g id=\"input-nodes\" fill=\"#8ce4b1\" font-family=\"system-ui, sans-serif\" font-size=\"12\">"
    )?;
    for node in 0..config.input_features() {
        let operand = if node < 7 { 'a' } else { 'b' };
        let label = format!("{operand}{}", node % 7);
        writeln!(
            svg,
            "<circle cx=\"{}\" cy=\"{:.2}\" r=\"4\"><title>Operand {operand}, bit {}</title></circle><text x=\"{}\" y=\"{:.2}\" text-anchor=\"end\">{label}</text>",
            x(0),
            y(node, 14),
            node % 7,
            x(0) - 12.0,
            y(node, 14) + 4.0
        )?;
    }
    writeln!(svg, "</g><g id=\"neurons\" fill=\"#e1eaf7\">")?;
    for (index, layer) in layers.iter().enumerate() {
        for output in 0..layer.outputs {
            let [a, b] = layer.parents(output);
            writeln!(
                svg,
                "<circle data-layer=\"{}\" data-node=\"{output}\" cx=\"{}\" cy=\"{:.2}\" r=\"{radius}\"><title>Layer {}, neuron {output}: parents {a} and {b}; one bias</title></circle>",
                index + 1,
                x(index + 1),
                y(output, layer.outputs),
                index + 1
            )?;
        }
    }
    writeln!(
        svg,
        "</g><g id=\"output-labels\" font-family=\"system-ui, sans-serif\" fill=\"#d6c2ff\" font-size=\"9\">"
    )?;
    for output in 0..config.modulus {
        writeln!(
            svg,
            "<text x=\"{}\" y=\"{:.2}\">{output}</text>",
            x(layers.len()) + 8.0,
            y(output, 97) + 3.0
        )?;
    }
    writeln!(svg, "</g></svg>")?;
    Ok(svg)
}

struct Stage {
    column: f64,
    count: usize,
    top: f64,
    height: f64,
    label: String,
    color: &'static str,
}

type Edge = (usize, usize, usize, usize, &'static str);

/// An explicit RMS computation: unary squares, two-parent sums, one scale
/// node, and two-parent normalized outputs. Constants are not activation nodes.
fn add_rms_tree(
    stages: &mut Vec<Stage>,
    edges: &mut Vec<Edge>,
    source: usize,
    label: String,
) -> usize {
    let width = stages[source].count;
    assert!(width.is_power_of_two());
    let (origin, top, height) = (
        stages[source].column,
        stages[source].top,
        stages[source].height,
    );
    let color = "#8197ac";
    let mut previous = stages.len();
    stages.push(Stage {
        column: origin + 0.1,
        count: width,
        top,
        height,
        label: "aux:square".into(),
        color,
    });
    for node in 0..width {
        edges.push((source, node, previous, node, color));
    }
    let mut count = width;
    let mut column = origin + 0.1;
    while count > 1 {
        count /= 2;
        column += 0.1;
        let current = stages.len();
        stages.push(Stage {
            column,
            count,
            top,
            height,
            label: "aux:binary sum".into(),
            color,
        });
        for node in 0..count {
            edges.push((previous, 2 * node, current, node, color));
            edges.push((previous, 2 * node + 1, current, node, color));
        }
        previous = current;
    }
    let scale = stages.len();
    stages.push(Stage {
        column: column + 0.1,
        count: 1,
        top,
        height,
        label: "aux:RMS scale with epsilon".into(),
        color,
    });
    edges.push((previous, 0, scale, 0, color));
    let output = stages.len();
    stages.push(Stage {
        column: origin + 1.2,
        count: width,
        top,
        height,
        label,
        color,
    });
    for node in 0..width {
        edges.push((source, node, output, node, color));
        edges.push((scale, 0, output, node, color));
    }
    output
}

/// Connectivity preview of shared linear Neighbor embeddings and their tied
/// transposed readout. The selected depth gives every logit access to every a/b
/// token. Connectivity does not assert that training will discover those paths.
fn draw_shared_neighbor() -> Result<String> {
    const WIDTH: usize = 128;
    const CLASSES: usize = 97;
    const EMBEDDING_DEPTH: usize = 32;
    const CORE_DEPTH: usize = 64;
    let config = SimpleConfig {
        architecture: Architecture::SharedNeighborMeshMlp,
        width: WIDTH,
        depth: CORE_DEPTH,
        embedding_mesh_depth: EMBEDDING_DEPTH,
        ..SimpleConfig::default()
    };
    config.validate()?;
    if !config
        .shared_neighbor_input_coverage()
        .iter()
        .all(|&count| count == 2 * CLASSES)
    {
        return Err("shared Neighbor diagram does not have complete input coverage".into());
    }
    let spacing = 34.0;
    let mut stages = Vec::<Stage>::new();
    let mut edges = Vec::<Edge>::new();
    let first_color = "#55c8f1";
    let second_color = "#f3ad6a";
    let shared_color = "#ccb6ff";
    let main_top = 180.0;
    let main_height = 1024.0;
    let mut branches = Vec::new();
    for (role, top) in [("a", 180.0), ("b", 756.0)] {
        let input = stages.len();
        stages.push(Stage {
            column: 0.0,
            count: WIDTH,
            top,
            height: 448.0,
            label: format!("{role} tokens"),
            color: "#8ce4b1",
        });
        let mut previous = input;
        for layer in 0..EMBEDDING_DEPTH {
            let current = stages.len();
            stages.push(Stage {
                column: (layer + 1) as f64,
                count: WIDTH,
                top,
                height: 448.0,
                label: format!("S{}", layer + 1),
                color: shared_color,
            });
            let shape = TwoParentLayer {
                inputs: WIDTH,
                outputs: WIDTH,
                stride: 1,
            };
            for node in 0..WIDTH {
                let [a, b] = shape.parents(node);
                edges.push((previous, a, current, node, first_color));
                edges.push((previous, b, current, node, second_color));
            }
            previous = current;
        }
        branches.push(previous);
    }
    let rotation = stages.len();
    stages.push(Stage {
        column: (EMBEDDING_DEPTH + 1) as f64,
        count: WIDTH,
        top: 756.0,
        height: 448.0,
        label: "R90".into(),
        color: "#f2a8d0",
    });
    for node in 0..WIDTH {
        let parent = if node.is_multiple_of(2) {
            node + 1
        } else {
            node - 1
        };
        edges.push((branches[1], parent, rotation, node, "#f2a8d0"));
    }
    let merged = stages.len();
    stages.push(Stage {
        column: (EMBEDDING_DEPTH + 2) as f64,
        count: WIDTH,
        top: main_top,
        height: main_height,
        label: "Join".into(),
        color: "#8ce4b1",
    });
    for node in 0..WIDTH {
        edges.push((branches[0], node, merged, node, "#8ce4b1"));
        edges.push((rotation, node, merged, node, "#f2a8d0"));
    }
    let mut previous = merged;
    for layer in 0..CORE_DEPTH {
        let normalized = add_rms_tree(&mut stages, &mut edges, previous, format!("N{}", layer + 1));
        let column = stages[normalized].column + 1.0;
        let current = stages.len();
        stages.push(Stage {
            column,
            count: WIDTH,
            top: main_top,
            height: main_height,
            label: format!("M{}", layer + 1),
            color: "#e1eaf7",
        });
        let shape = TwoParentLayer {
            inputs: WIDTH,
            outputs: WIDTH,
            stride: 1,
        };
        for node in 0..WIDTH {
            let [a, b] = shape.parents(node);
            edges.push((normalized, a, current, node, first_color));
            edges.push((normalized, b, current, node, second_color));
        }
        let added = stages.len();
        stages.push(Stage {
            column: column + 1.0,
            count: WIDTH,
            top: main_top,
            height: main_height,
            label: "+".into(),
            color: "#8ce4b1",
        });
        for node in 0..WIDTH {
            edges.push((previous, node, added, node, "#8ce4b1"));
            edges.push((current, node, added, node, "#8ce4b1"));
        }
        previous = added;
    }
    previous = add_rms_tree(&mut stages, &mut edges, previous, "Final N".into());
    for layer in 0..EMBEDDING_DEPTH {
        let current = stages.len();
        let shared = EMBEDDING_DEPTH - layer;
        let column = stages[previous].column + 1.0;
        stages.push(Stage {
            column,
            count: WIDTH,
            top: main_top,
            height: main_height,
            label: format!("S{shared} R"),
            color: shared_color,
        });
        let shape = TwoParentLayer {
            inputs: WIDTH,
            outputs: WIDTH,
            stride: WIDTH - 1,
        };
        for node in 0..WIDTH {
            let [a, b] = shape.parents(node);
            edges.push((previous, a, current, node, first_color));
            edges.push((previous, b, current, node, second_color));
        }
        previous = current;
    }
    let x = |stage: &Stage| 88.0 + spacing * stage.column;
    let y = |stage: &Stage, node: usize| {
        stage.top + (node as f64 + 0.5) * stage.height / stage.count as f64
    };
    let width = x(stages.last().unwrap()) + 85.0;
    let height = 1260.0;
    let parameters = config.parameter_count();
    let mut incoming = vec![Vec::<String>::new(); stages.len() * WIDTH];
    for &(source, parent, target, node, _) in &edges {
        incoming[target * WIDTH + node].push(format!("stage {source} node {parent}"));
    }
    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\" role=\"img\" aria-labelledby=\"title desc\">\n\
         <title id=\"title\">Shared Neighbor embeddings and transposed Neighbor readout</title>\n\
         <desc id=\"desc\">Full connectivity: shared linear Neighbor embeddings, Rotate90 and sum, 64 pre-RMSNorm Neighbor ReLU residual blocks, final RMSNorm and the shared transposed Neighbor readout. Each RMSNorm is expanded into unary squares, a binary sum tree, a fixed scale and two-parent output multiplications. All nodes have at most two incoming activation links. The first 97 terminal values are answer logits. Every output reaches every a and b token.</desc>\n\
         <rect width=\"100%\" height=\"100%\" fill=\"#0a111c\"/>\n\
         <g font-family=\"system-ui, sans-serif\" fill=\"#edf3fb\">\n\
         <text x=\"30\" y=\"38\" font-size=\"25\">Shared Neighbor embedding + Neighbor readout</text>\n\
         <text x=\"30\" y=\"67\" font-size=\"14\" fill=\"#a9b6c8\">Full connectivity · 128 channels · 32 embedding layers · 64 core layers · 97 answers</text>\n\
         <text x=\"30\" y=\"91\" font-size=\"13\" fill=\"#a9b6c8\">{} neurons · {} connections · {parameters} unique parameters · all connections shown</text>\n\
         <text x=\"30\" y=\"115\" font-size=\"13\" fill=\"{shared_color}\">S1–S32 are the same weights for a, b and readout. R marks the transposed readout; M marks the ReLU core.</text>\n\
         <text x=\"30\" y=\"139\" font-size=\"12\" fill=\"#a9b6c8\">N: pre-RMSNorm with binary trees. M: Neighbor + ReLU with bias. +: residual sum. Final N: RMSNorm before readout. Maximum two incoming links.</text>\n\
         <path d=\"M30 157h23\" stroke=\"{first_color}\" stroke-width=\"2\"/><text x=\"62\" y=\"161\" font-size=\"12\">First link</text>\n\
         <path d=\"M165 157h23\" stroke=\"{second_color}\" stroke-width=\"2\"/><text x=\"197\" y=\"161\" font-size=\"12\">Second link</text>\n\
         <text x=\"325\" y=\"161\" font-size=\"12\" fill=\"#a9b6c8\">All 97 outputs reach all 194 a/b indicators. Left to right; zoom to inspect.</text></g>\n\
         <g id=\"connections\" fill=\"none\" stroke-width=\"0.6\" stroke-opacity=\"0.4\">\n",
        stages.iter().map(|stage| stage.count).sum::<usize>(),
        edges.len()
    );
    for (source, parent, target, node, color) in &edges {
        writeln!(
            svg,
            "<path data-source-stage=\"{source}\" data-parent=\"{parent}\" data-target-stage=\"{target}\" data-node=\"{node}\" d=\"M{:.2} {:.2}L{:.2} {:.2}\" stroke=\"{color}\"/>",
            x(&stages[*source]),
            y(&stages[*source], *parent),
            x(&stages[*target]),
            y(&stages[*target], *node)
        )?;
    }
    writeln!(
        svg,
        "</g><g id=\"stages\" font-family=\"system-ui, sans-serif\">"
    )?;
    for (index, stage) in stages.iter().enumerate() {
        let label_y = if stage.top == main_top {
            173.0
        } else {
            stage.top - 12.0
        };
        if !stage.label.starts_with("aux:") {
            writeln!(
                svg,
                "<text x=\"{}\" y=\"{label_y}\" font-size=\"9\" text-anchor=\"middle\" fill=\"{}\">{}</text>",
                x(stage),
                stage.color,
                stage.label
            )?;
        }
        for node in 0..stage.count {
            let input = stage.column == 0.0;
            let unused_output = index + 1 == stages.len() && node >= CLASSES;
            let padded = input && node >= CLASSES;
            let color = if padded || unused_output {
                "#4c5b70"
            } else {
                stage.color
            };
            let radius = if stage.label.starts_with("aux:") {
                0.8
            } else if stage.height < main_height {
                1.3
            } else {
                2.2
            };
            let parents = &incoming[index * WIDTH + node];
            let title = if padded {
                "Fixed zero padding".into()
            } else if input {
                format!("{}: token {node} indicator", stage.label)
            } else {
                format!(
                    "{}, node {node}; inputs: {}",
                    stage.label.trim_start_matches("aux:"),
                    parents.join(" and ")
                )
            };
            writeln!(
                svg,
                "<circle data-stage=\"{index}\" data-node=\"{node}\" cx=\"{}\" cy=\"{:.2}\" r=\"{radius}\" fill=\"{color}\"><title>{title}</title></circle>",
                x(stage),
                y(stage, node)
            )?;
            if input && (node.is_multiple_of(8) || node == CLASSES - 1) {
                let label = if padded {
                    "0 pad".into()
                } else {
                    node.to_string()
                };
                writeln!(
                    svg,
                    "<text x=\"{}\" y=\"{:.2}\" font-size=\"8\" text-anchor=\"end\" fill=\"#8ce4b1\">{label}</text>",
                    x(stage) - 7.0,
                    y(stage, node) + 2.5
                )?;
            }
            if index + 1 == stages.len() && node < CLASSES {
                writeln!(
                    svg,
                    "<text x=\"{}\" y=\"{:.2}\" font-size=\"9\" fill=\"{shared_color}\">{node}</text>",
                    x(stage) + 8.0,
                    y(stage, node) + 3.0
                )?;
            }
        }
    }
    writeln!(svg, "</g></svg>")?;
    Ok(svg)
}

fn main() -> Result<()> {
    let mut arguments = std::env::args().skip(1);
    let directory = arguments
        .next()
        .map_or_else(|| PathBuf::from("docs/diagrams"), PathBuf::from);
    if arguments.next().is_some() {
        return Err("usage: full-mesh-svg [OUTPUT_DIRECTORY]".into());
    }
    fs::create_dir_all(&directory)?;
    for (connectivity, depth) in [(Connectivity::Butterfly, 8), (Connectivity::Neighbor, 64)] {
        let config = SimpleConfig {
            architecture: Architecture::FullMeshMlp,
            encoding: Encoding::Binary,
            width: 128,
            depth,
            connectivity,
            ..SimpleConfig::default()
        };
        let path = directory.join(format!("full-mesh-{connectivity}.svg"));
        fs::write(&path, draw(config)?)?;
        println!("{}", path.display());
    }
    let path = directory.join("shared-neighbor-embedding-readout.svg");
    fs::write(&path, draw_shared_neighbor()?)?;
    println!("{}", path.display());
    Ok(())
}
