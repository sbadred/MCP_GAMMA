#include <iostream>
#include <vector>
#include <string>
#include "CheckGamma.h"

std::vector<std::string> generateCompletePool(int QubitNumber) {
    std::vector<std::string> CompleteGenerators;

    for (int i = 0; i < QubitNumber; ++i) {
        if (i == 0) {
            CompleteGenerators.push_back("Y");
        }
        else if (i == 1) {
            CompleteGenerators[0] = "Z" + CompleteGenerators[0];
            CompleteGenerators.push_back("YI");
        }
        else {
            for (size_t j = 0; j < CompleteGenerators.size(); ++j) {
                CompleteGenerators[j] = "Z" + CompleteGenerators[j];
            }
            CompleteGenerators.push_back("Y" + std::string(i, 'I'));
            CompleteGenerators.push_back("IY" + std::string(i - 1, 'I'));
        }
    }

    return CompleteGenerators;
}

int main() {
    // =========================================================================
    // N = 16 Qubits (H8 System).
    // Pool size = 28 operators (2N - 4 = 28).
    // =========================================================================
    int n = 16;

    // --- 1. THEORETICAL COMPLETE POOL ---
    std::vector<std::string> Pauli_1 = generateCompletePool(n);
    MCP mcp_1(Pauli_1, n);

    std::cout << "Processing matrices on GPU (Theoretical Pool)..." << std::endl;
    mcp_1.generateF2Mat();
    mcp_1.reduceF2Mat();
    mcp_1.generateDecimalPairs();
    mcp_1.generateProductGroup();

    int closureSize_1 = mcp_1.getBracketClosureSize();

    std::cout << "\n==========================================" << std::endl;
    std::cout << "RESULT: The size of the theoretical bracket closure is: " << closureSize_1 << std::endl;
    std::cout << "==========================================\n" << std::endl;


    // --- 2. HARDCODED H8 POOL ---
    std::vector<std::string> Pauli_2 = {
        "YXIIIIIIYYIIIIII",
        "IIYXIIIIYYIIIIII",
        "IIIIYXIIYYIIIIII",
        "IIIIIIYXYYIIIIII",
        "IIIIYIXIYIYIIIII",
        "IIIYXIIIIYYIIIII",
        "IIIIIYXIIYYIIIII",
        "IIYIIXIIYIIYIIII",
        "IIIIYIIXYIIYIIII",
        "YXIIIIIIIIYYIIII",
        "IIIYIIXIIYIIYIII",
        "IIIYXIIIIIIYYIII",
        "IIYIIIIXYIIIIYII",
        "IYIIIIXIIYIIIIYI",
        "YIIIIIIXYIIIIIIY",
        "YXIIIIIIIIIIIIYY",
        "IYIIXIIIYYIIIIII",
        "YIXIIIIIYIYIIIII",
        "IIIIIZXZXXIIZIZY",
        "IIIIIYIYZXIIIYII",
        "IIIIIXXZYXIZZZII",
        "IIIIIXIIIXXZYIII",
        "IIIIIZXZIZZZIXXY",
        "IIIIIZIZYZYYZXII",
        "IIIIIXXIZIZIZZXY",
        "IIIIIYIXZIZZIZII",
        "IIIIIIYZYIZIZXZY",
        "IIIIIIIXXXIIZIYI"
    };

    std::cout << "--- Evaluating Bracket Closure (H8 Pool) ---" << std::endl;
    std::cout << "Number of qubits (N): " << n << std::endl;
    std::cout << "Pool size (Rank): " << Pauli_2.size() << "\n" << std::endl;

    MCP mcp_2(Pauli_2, n);

    std::cout << "Processing matrices on GPU (H8 Pool)..." << std::endl;
    mcp_2.generateF2Mat();
    mcp_2.reduceF2Mat();
    mcp_2.generateDecimalPairs();
    mcp_2.generateProductGroup();

    int full_mcp_size = mcp_2.getBracketClosureSize();

    std::cout << "\n==========================================" << std::endl;
    std::cout << "RESULT: The size of the H8 bracket closure is: " << full_mcp_size << std::endl;
    std::cout << "==========================================\n" << std::endl;


    // --- 3. PROVE MINIMALITY (Leave-one-out testing) ---
    std::cout << std::string(50, '-') << std::endl;
    std::cout << "\n4. Prove Minimality (Remove one element)\n" << std::endl;

    for (size_t i = 0; i < Pauli_2.size(); ++i) {
        std::string element_to_remove = Pauli_2[i];

        // Create a copy and erase the i-th element safely
        std::vector<std::string> reduced_MCP = Pauli_2;
        reduced_MCP.erase(reduced_MCP.begin() + i);

        std::cout << "Testing minimality by removing: '" << element_to_remove << "'" << std::endl;

        // Create a fresh MCP object on the GPU for the reduced pool
        MCP mcp_reduced(reduced_MCP, n);
        mcp_reduced.generateF2Mat();
        mcp_reduced.reduceF2Mat();
        mcp_reduced.generateDecimalPairs();
        mcp_reduced.generateProductGroup();

        int reduced_mcp_size = mcp_reduced.getBracketClosureSize();

        std::cout << "Size of generated Lie algebra (Reduced MCP): " << reduced_mcp_size << std::endl;

        // Print the proof conclusion
        if (reduced_mcp_size < full_mcp_size) {
            std::cout << "element: " << i << std::endl;
            std::cout << "\nSUCCESS: Removing just one element dropped the algebra size by "
                      << (full_mcp_size - reduced_mcp_size) << " operators!" << std::endl;
            std::cout << "This proves the removed element was mathematically necessary and the pool is strictly minimal.\n" << std::endl;
            std::cout << std::string(50, '-') << std::endl;
        } else {
            std::cout << "\nWARNING: The algebra size did not drop! The pool is NOT minimal.\n" << std::endl;
            std::cout << std::string(50, '-') << std::endl;
        }
    }

    return 0;
}