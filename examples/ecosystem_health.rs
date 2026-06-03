//! CLI binary that prints the SuperInstance ecosystem health report.
//!
//! Run with: cargo run --example ecosystem_health

use conservation_spectral_topology::ecosystem::{
    verify_ecosystem, print_health_report, Agent, AgentType,
};
use nalgebra::{DMatrix, DVector};

fn main() {
    let agents = vec![
        Agent {
            name: "lever-runner".into(),
            agent_type: AgentType::Execution,
            budget: 100.0,
            current_usage: 72.0,
        },
        Agent {
            name: "pincherOS".into(),
            agent_type: AgentType::Memory,
            budget: 80.0,
            current_usage: 65.0,
        },
        Agent {
            name: "PLATO".into(),
            agent_type: AgentType::Intelligence,
            budget: 150.0,
            current_usage: 142.0,
        },
        Agent {
            name: "agent-A".into(),
            agent_type: AgentType::Identity,
            budget: 60.0,
            current_usage: 30.0,
        },
        Agent {
            name: "agent-B".into(),
            agent_type: AgentType::Identity,
            budget: 60.0,
            current_usage: 45.0,
        },
        Agent {
            name: "agent-C".into(),
            agent_type: AgentType::Identity,
            budget: 60.0,
            current_usage: 25.0,
        },
    ];

    let adjacency = DMatrix::from_row_slice(
        6,
        6,
        &[
            0.0,  1.0,  1.0,  0.0,  0.0,  0.0,
            1.0,  0.0,  1.0,  1.0,  0.0,  0.0,
            1.0,  1.0,  0.0,  1.0,  1.0,  1.0,
            0.0,  1.0,  1.0,  0.0,  1.0,  0.0,
            0.0,  0.0,  1.0,  1.0,  0.0,  1.0,
            0.0,  0.0,  1.0,  0.0,  1.0,  0.0,
        ],
    );

    let flow = DVector::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

    let report = verify_ecosystem(&agents, &adjacency, &flow);
    print_health_report(&report);
}
