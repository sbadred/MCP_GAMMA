#include <iostream>
#include <vector>
#include <string>
#include "CheckGamma.h" // Contains Yaromir's MCP class, pOp struct, and int2pauli

typedef unsigned long pOp;
enum pauli_e {pI, pX, pY, pZ};

std::string int2pauli_local(pOp iiTab, int nQubit) {
    std::string ipauli = "";
    for (int q = 0; q < nQubit; ++q) {
        int id = (iiTab >> (q << 1) & 3);

        if (id == pI) {
            ipauli += 'I';
        } else if (id == pX) {
            ipauli += 'X';
        } else if (id == pY) {
            ipauli += 'Y';
        } else if (id == pZ) {
            ipauli += 'Z';
        }
    }
    return ipauli;
}

std::vector<std::string> int2pauli(const pOp* Tab, int nQubit, uint incMCP) {
    std::vector<std::string> pool;
    for (size_t i = 0; i < incMCP; ++i) {
        std::string ipauli = int2pauli_local(Tab[i], nQubit);
        pool.push_back(ipauli);
    }
    return pool;
}

// =========================================================================
// VERSION 1: (Accepts low-level pOp array)
// =========================================================================
int check_bracket(const pOp* mcp, const uint n, const uint nQubit) {
    std::vector<std::string> MCP_string = int2pauli(mcp, nQubit, n);
    MCP mcp_ = MCP(MCP_string, nQubit);
    mcp_.generateF2Mat();
    mcp_.reduceF2Mat();
    mcp_.generateGamma();

    int value = mcp_.checkBracketIndep_opt();
    return (value != 0) ? 1 : 0;
}

// =========================================================================
// VERSION 2:  (Accepts std::vector<std::string>)
// =========================================================================
int check_bracket(const std::vector<std::string>& mcp_strings, const uint nQubit) {
    // Skip the int2pauli conversion since we already have strings!
    MCP mcp_ = MCP(mcp_strings, nQubit);
    mcp_.generateF2Mat();
    mcp_.reduceF2Mat();
    mcp_.generateGamma();

    int value = mcp_.checkBracketIndep_opt();
    return (value != 0) ? 1 : 0;
}

int main() {
    // =========================================================================
    // System: H8 Molecule (N = 16 Qubits)
    // Target Rank for completeness: 2N - 4 = 28
    // =========================================================================
    int nQubit = 16;
    int target_rank = (2 * nQubit) - 4;

    std::vector<std::string> Pauli_H8 = {
        "YXIIIIIIYYIIIIII", "IIYXIIIIYYIIIIII", "IIIIYXIIYYIIIIII",
        "IIIIIIYXYYIIIIII", "IIIIYIXIYIYIIIII", "IIIYXIIIIYYIIIII",
        "IIIIIYXIIYYIIIII", "IIYIIXIIYIIYIIII", "IIIIYIIXYIIYIIII",
        "YXIIIIIIIIYYIIII", "IIIYIIXIIYIIYIII", "IIIYXIIIIIIYYIII",
        "IIYIIIIXYIIIIYII", "IYIIIIXIIYIIIIYI", "YIIIIIIXYIIIIIIY",
        "YXIIIIIIIIIIIIYY", "IYIIXIIIYYIIIIII", "YIXIIIIIYIYIIIII",
        "IIIIIZXZXXIIZIZY", "IIIIIYIYZXIIIYII", "IIIIIXXZYXIZZZII",
        "IIIIIXIIIXXZYIII", "IIIIIZXZIZZZIXXY", "IIIIIZIZYZYYZXII",
        "IIIIIXXIZIZIZZXY", "IIIIIYIXZIZZIZII", "IIIIIIYZYIZIZXZY",
        "IIIIIIIXXXIIZIYI"
    };

    size_t num_terms = Pauli_H8.size();

    std::cout << "The number of terms in the Pauli_H8 pool is: " << num_terms << std::endl;

    std::cout << "==========================================================" << std::endl;
    std::cout << "  MCP VERIFICATION: RANK CONDITION & BRACKET INDEPENDENCE " << std::endl;
    std::cout << "==========================================================\n" << std::endl;

    // --- STEP 1: VERIFY THE RANK CONDITION ---
    std::cout << "--- STEP 1: Evaluating Gamma Matrix Rank ---" << std::endl;
    MCP mcp(Pauli_H8, nQubit);
    mcp.generateF2Mat();
    mcp.generateGamma();

    int computed_rank = mcp.GammaRank(); // Ensure this matches Yaromir's exact method name!

    std::cout << "Target Theoretical Rank: " << target_rank << std::endl;
    std::cout << "Computed Gamma Rank:     " << computed_rank << std::endl;

    if (computed_rank == target_rank) {
        std::cout << "[SUCCESS] The pool strictly satisfies the MCP rank condition.\n" << std::endl;
    } else {
        std::cout << "[FAILURE] The rank condition is not met.\n" << std::endl;
        return 1;
    }

    // --- STEP 2: VERIFY BRACKET INDEPENDENCE ---
    std::cout << "--- STEP 2: Evaluating Bracket Independence ---" << std::endl;
    std::cout << "Checking if the pool is strictly bracket-independent..." << std::endl;

    // Now safely calls Version 2 of the function!
    int is_independent = check_bracket(Pauli_H8, nQubit);

    std::cout << "\n==========================================================" << std::endl;
    if (is_independent) {
        std::cout << "CONCLUSION: The pool is COMPLETE (Rank = 2N-4) and \n"
                  << "strictly BRACKET-INDEPENDENT. It is a Minimal Complete Pool." << std::endl;
    } else {
        std::cout << "CONCLUSION: The pool contains redundant elements and is NOT minimal." << std::endl;
    }
    std::cout << "==========================================================\n" << std::endl;

    return 0;
}