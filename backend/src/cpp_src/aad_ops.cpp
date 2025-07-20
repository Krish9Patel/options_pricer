#include "aad_ops.h"

std::shared_ptr<Tape> curr_tape = nullptr;

void initialize_aad_tape() {
    if (!curr_tape) {
        curr_tape = std::make_shared<Tape>();
    } else {
        curr_tape -> clear(); // Clear existing tape if initialized
    }
}

// Perform reverse pass to compute adjoints
void reverse_pass(int start_idx) {
    if (!curr_tape || curr_tape -> nodes.empty()) {
        return;
    }

    if (start_idx != -1 && start_idx < curr_tape -> nodes.size()) {
        curr_tape -> nodes[start_idx].adjoint = 1.0; // Initialize adjoint of the output node
    } else {
        curr_tape -> nodes.back().adjoint = 1.0; // Default to last node if no start index provided
    }

    for (int i = curr_tape -> nodes.size() - 1; i >=0; --i) {
        Node& node = curr_tape -> nodes[i];

        if (node.adjoint == 0.0 && i != start_idx) {
            continue; // Skip if adjoint = 0 
        }

        for (size_t p_idx = 0; p_idx < node.parents.size(); ++p_idx) {
            int parent_tape_idx = node.parents[p_idx];
            double weight = node.weights[p_idx];
            current_tape -> nodes[parent_tape_idx].adjoint += node.adjoint * weight;
        }
        node.adjoint = 0.0; 
    }
}
