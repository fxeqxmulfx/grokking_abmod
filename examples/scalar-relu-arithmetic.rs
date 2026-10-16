//! Constructive representability control, not an optimizer or Kaiming-trained model.
//! A plain 2 -> 128 x 8 -> 1 ReLU MLP computes both modular rules exactly.
use grokking_abmod::{
    Result,
    gpt::{Equation, EquationDataset, ModularOperation},
    simple::{Dense, GpuRegressionMlp, RegressionConfig},
};

const MODULUS: usize = 97;

struct Affine {
    shape: Dense,
    parameters: Vec<f32>,
}

impl Affine {
    fn new(shape: Dense) -> Self {
        Self {
            shape,
            parameters: vec![0.0; shape.parameters() + shape.outputs],
        }
    }

    fn add(&mut self, input: usize, output: usize, weight: f32) {
        self.parameters[input * self.shape.outputs + output] += weight;
    }

    fn bias(&mut self, output: usize, value: f32) {
        self.parameters[self.shape.parameters() + output] = value;
    }
}

fn constructed(config: RegressionConfig, operation: ModularOperation) -> Vec<f32> {
    let mut layers: Vec<_> = config.layers().into_iter().map(Affine::new).collect();

    // H1: a, b and the 95 integer-spaced hinges needed to interpolate inverse(b).
    layers[0].add(0, 0, 1.0);
    layers[0].add(1, 1, 1.0);
    for knot in 1..96 {
        layers[0].add(1, knot + 1, 1.0);
        layers[0].bias(knot + 1, -(knot as f32));
    }
    let mut operand = vec![0.0; config.width];
    if operation == ModularOperation::Divide {
        let values: Vec<_> = (0..97)
            .map(|b| {
                if b == 0 {
                    0
                } else {
                    (1..97).find(|inverse| inverse * b % 97 == 1).unwrap()
                }
            })
            .collect();
        let slopes: Vec<_> = values.windows(2).map(|pair| pair[1] - pair[0]).collect();
        operand[1] = slopes[0] as f32;
        for knot in 1..96 {
            operand[knot + 1] = (slopes[knot] - slopes[knot - 1]) as f32;
        }
    } else {
        operand[1] = 1.0;
    }

    // H2: s=a+t and the two nonnegative parts of a-t, where t=b or inverse(b).
    layers[1].add(0, 0, 1.0);
    layers[1].add(0, 1, 1.0);
    layers[1].add(0, 2, -1.0);
    for (input, coefficient) in operand.into_iter().enumerate() {
        layers[1].add(input, 0, coefficient);
        layers[1].add(input, 1, -coefficient);
        layers[1].add(input, 2, coefficient);
    }

    // Integer square identity: x^2 = x + 2*sum_{k=1}^{191} ReLU(x-k).
    // Split its 286 useful hinges over three layers and carry a positive sum.
    // The resulting product is (s^2 - |a-t|^2)/4, exactly at integer operands.
    layers[2].add(0, 0, 1.0);
    layers[2].add(1, 1, 1.0);
    layers[2].add(2, 1, 1.0);
    layers[2].add(0, 2, 0.25);
    layers[2].add(1, 2, -0.25);
    layers[2].add(2, 2, -0.25);
    for knot in 1..=120 {
        layers[2].add(0, knot + 2, 1.0);
        layers[2].bias(knot + 2, -(knot as f32));
    }

    layers[3].add(0, 0, 1.0);
    layers[3].add(1, 1, 1.0);
    layers[3].add(2, 2, 1.0);
    for input in 3..123 {
        layers[3].add(input, 2, 0.5);
    }
    for knot in 121..=191 {
        let output = knot - 121 + 3;
        layers[3].add(0, output, 1.0);
        layers[3].bias(output, -(knot as f32));
    }
    for knot in 1..=49 {
        let output = knot - 1 + 74;
        layers[3].add(1, output, 1.0);
        layers[3].bias(output, -(knot as f32));
    }

    layers[4].add(2, 0, 1.0);
    for input in 3..74 {
        layers[4].add(input, 0, 0.5);
    }
    for input in 74..123 {
        layers[4].add(input, 0, -0.5);
    }
    for knot in 50..=95 {
        layers[4].add(1, knot - 49, 1.0);
        layers[4].bias(knot - 49, -(knot as f32));
    }

    // H6: product z and the first 60 pairs of hinges for its mod-97 sawtooth.
    for output in 0..121 {
        layers[5].add(0, output, 1.0);
        for input in 1..47 {
            layers[5].add(input, output, -0.5);
        }
    }
    for multiple in 1..=60 {
        layers[5].bias(2 * multiple - 1, -((97 * multiple - 1) as f32));
        layers[5].bias(2 * multiple, -((97 * multiple) as f32));
    }

    // At integer z, ReLU(z-(97q-1))-ReLU(z-97q) equals 1 iff z >= 97q.
    // Subtract the first 60 steps, then the last 35; partial sums stay positive.
    layers[6].add(0, 0, 1.0);
    for multiple in 1..=60 {
        layers[6].add(2 * multiple - 1, 0, -97.0);
        layers[6].add(2 * multiple, 0, 97.0);
    }
    for multiple in 61..=95 {
        let output = 2 * (multiple - 61) + 1;
        layers[6].add(0, output, 1.0);
        layers[6].bias(output, -((97 * multiple - 1) as f32));
        layers[6].add(0, output + 1, 1.0);
        layers[6].bias(output + 1, -((97 * multiple) as f32));
    }

    // H8 is a nonnegative identity so the control matches a planned depth.
    for channel in 0..config.width {
        layers[7].add(channel, channel, 1.0);
    }
    layers[8].add(0, 0, 1.0);
    for multiple in 61..=95 {
        let input = 2 * (multiple - 61) + 1;
        layers[8].add(input, 0, -97.0);
        layers[8].add(input + 1, 0, 97.0);
    }
    layers
        .into_iter()
        .flat_map(|layer| layer.parameters)
        .collect()
}

fn evaluate(model: &mut GpuRegressionMlp, equations: &[Equation]) -> Result<(usize, f64)> {
    let features: Vec<_> = equations
        .iter()
        .flat_map(|e| [e.a as f32, e.b as f32])
        .collect();
    let targets: Vec<_> = equations.iter().map(|e| e.answer as f32).collect();
    let table = model.upload_table(&features, &targets)?;
    let predictions = model.predictions(&table)?;
    let correct = predictions
        .iter()
        .zip(&targets)
        .filter(|(actual, expected)| actual == expected)
        .count();
    let maximum_error = predictions
        .iter()
        .zip(targets)
        .map(|(&actual, expected)| f64::from((actual - expected).abs()))
        .fold(0.0, f64::max);
    Ok((correct, maximum_error))
}

fn run() -> Result<()> {
    let config = RegressionConfig {
        width: 128,
        depth: 8,
        pre_rmsnorm: false,
        residual: false,
    };
    for operation in [ModularOperation::Divide, ModularOperation::Multiply] {
        let parameters = constructed(config, operation);
        let mut model = GpuRegressionMlp::from_parameters(0, config, &parameters)?;
        let dataset = EquationDataset::new(MODULUS, operation, 0.5, 11)?;
        let train = evaluate(&mut model, dataset.split(false))?;
        let held_out = evaluate(&mut model, dataset.split(true))?;
        let total = dataset.split(false).len() + dataset.split(true).len();
        if train.0 + held_out.0 != total || train.1 != 0.0 || held_out.1 != 0.0 {
            return Err(format!(
                "constructive {operation} control failed: train={train:?}, held_out={held_out:?}"
            )
            .into());
        }
        println!(
            "{{\"control\":\"constructed_weights_not_trained\",\"operation\":\"{operation}\",\"modulus\":97,\"width\":128,\"depth\":8,\"parameters\":{},\"inputs\":2,\"outputs\":1,\"scale\":\"raw\",\"trainCorrect\":{},\"trainRows\":{},\"heldoutCorrect\":{},\"heldoutRows\":{},\"maxAbsoluteError\":0,\"passed\":true}}",
            config.parameter_count(),
            train.0,
            dataset.split(false).len(),
            held_out.0,
            dataset.split(true).len()
        );
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
