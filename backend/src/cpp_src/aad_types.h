#ifndef AAD_TYPES_H
#define AAD_TYPES_H

#include <vector>
#include <cstdint>

struct Node {
    double  value; // Forward pass value
    double  adjoint; // Backward pass adjoint (∂output/∂this)
    double  weight_1; // ∂(this)/∂(parent_1)
    double  weight_2; // ∂(this)/∂(parent_2)
    int32_t parent_1; // Index of first parent (-1 if leaf)
    int32_t parent_2; // Index of second parent (-1 if unary)
};

static_assert(sizeof(Node) == 40, "Node should pack to 40 bytes");

class Tape {
public:
    static Tape& instance() {
        static thread_local Tape tape;
        return tape;
    }

    void reserve(size_t num_nodes) { nodes_.reserve(num_nodes); }
    void clear() { nodes_.clear(); }
    size_t size() const { return nodes_.size(); }

    // Push a new node, return its index.
    size_t push_back(const Node& node) {
        size_t idx = nodes_.size();
        nodes_.push_back(node);
        return idx;
    }

    Node& operator[](size_t idx) { return nodes_[idx]; }
    const Node& operator[](size_t idx) const { return nodes_[idx]; }

    // Backward pass: propagate adjoints in reverse topological order
    // (which is reverse insertion order, since we only ever reference
    // parents with smaller indices).
    void propagate() {
        for (int i = static_cast<int>(nodes_.size()) - 1; i >= 0; --i) {
            const Node& node = nodes_[i];
            const double adj = node.adjoint;
            if (adj == 0.0) continue; // sparse-adjoint short-circuit
            if (node.parent_1 >= 0) {
                nodes_[node.parent_1].adjoint += adj * node.weight_1;
            }
            if (node.parent_2 >= 0) {
                nodes_[node.parent_2].adjoint += adj * node.weight_2;
            }
        }
    }

    void reset_adjoints() {
        for (auto& node : nodes_) node.adjoint = 0.0;
    }

private:
    std::vector<Node> nodes_;
};

// Automatic differentiation enabled double
class AD_double {
public:
    AD_double() : val_(0.0), idx_(-1) {}
    explicit AD_double(double v) : val_(v), idx_(-1) {} // constant by default

    // Mark this variable as an input for differentiation.
    // To note: this mutates idx_, so the caller cannot declare AD_double inputs const.
    void mark_as_input() {
        idx_ = static_cast<int>(Tape::instance().push_back(Node{
            val_, // Value
            0.0, // Adjoint
            0.0, // Weight 1
            0.0, // Weight 2
            -1, // Parent 1
            -1 // Parent 2
        }));
    }

    double val() const { return val_; }
    double adjoint() const {
        return (idx_ >= 0) ? Tape::instance()[idx_].adjoint : 0.0;
    }

    void set_val(double v) { val_ = v; }
    int tape_index() const { return idx_; }
    void set_tape_index(int i) { idx_ = i; }

    // Seed the adjoint = 1.0 at this node, then run the backward pass.
    void propagate_adjoint(double seed = 1.0) const {
        if (idx_ >= 0) {
            Tape::instance().reset_adjoints();
            Tape::instance()[idx_].adjoint = seed;
            Tape::instance().propagate();
        }
    }

private:
    double val_;
    int idx_; // index into tape (-1 = constant, not on tape)
};

#endif
