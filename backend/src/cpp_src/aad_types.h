#ifndef AAD_TYPES_H
#define AAD_TYPES_H

#include <vector>
#include <map>
#include <memory>

// Node structure for the AAD computation graph aka the Tape
struct Node {
    double value;
    double adjoint;
    int op_type;
    std::vector<int> parents;
    std::vector<double> weights;

    Node() : value(0.0), adjoint(0.0), op_type(0.0) {} // Default constructor
};

class Tape {
    public:
        std::vector<Node> nodes;
        int curr_idx;

        Tape() : curr_idx(0) {}

        // Add new node to the Tape, return current index, iterate index
        int record(double val, int op_type = 0, const std::vector<int>& parents = {}, const std::vector<double>& weights = {}) {
            nodes.emplace_back();
            nodes[curr_idx].value = val;
            nodes[curr_idx].op_type = op_type;
            nodes[curr_idx].parents = parents;
            nodes[curr_idx].weights = weights;
            return curr_idx++;
        }

        void clear() {
            nodes.clear();
            curr_idx = 0;
        }
};

// Global tape instance to make things simpler (manage in better way later)
extern std::shared_ptr<Tape> curr_tape;

// Automatic differentiation enabled double
class AD_double {
    public:
        int tape_idx = 0; // Index of var value on global tape

        AD_double(double val = 0.0) {
            if (curr_tape) {
                tape_idx = curr_tape -> record(val);
            } else {
                // No tape active then treat as simple double conceptually (handle more explicitly later, ie error or deffered tape setup)
                tape_idx = -1; // Indicates no tape 
            }
        }

        // Constuctor for val already on tape
        AD_double(int idx) : tape_idx(idx) {}

        // Get primal val
        double val() const{
            if (curr_tape && tape_idx != -1) {
                return curr_tape -> nodes[tape_idx].value;
            }
            return 0.0; // Possibly change to throw error ltr if not associated with tape
        }
        
        
        // Get adjoint val
        double adj() const {
            if (curr_tape && tape_idx != -1) {
                return curr_tape -> nodes[tape_idx].adjoint;
            }
        }

        // Setter will be used during reverse pass initialization
        void set_adjoint(double adj) {
            if (curr_tape && tape_idx != -1) {
                curr_tape -> nodes[tape_idx].adjoint = adj;
            }
        }

};

// Function to initialize the global tape
void initialize_aad_tape();

// Function to perform reverse pass and compute adjoints
void reverse_pass(int start_idx);

#endif
