/*
 * Original Author: Yaromir
*/
#include<iostream>
#include<vector>
#include<string>
#include<fstream>
#include<iomanip>
#include<cstring>
#include<math.h>
#include<stdio.h>
#include<bitset>
#include<algorithm>



#include <cuda_runtime.h>

std::vector<std::string> readMcpTxt(const char* filePath);
int* phi(std::string pauliString, int n);
std::string phiInverse(int* f2Rep, int order);
int* hostMcpVecToDevMcpMat(std::vector<std::string> mcpVec, int n);
int* f2DevMatmul(int* devA, int* devB, int rowsA, int colsA, int rowsB, int colsB) ;
std::pair<int*, int> f2RREF(int* devA, int rowsA, int colsA) ;
int cardinalityOfProductGroup(int rank);
int* devGenerateGamma(int* devA, int rowsA, int colsA);
std::pair<uint32_t*, uint32_t*> devF2MatToDecimalPairs(int* devA, int rowsA, int colsA);
int* devFindPivotCols(int* devRrefA, int rowsA, int colsA, int rank);
std::pair<int*, int*> devFindPivotAndNhostMcpVecToDevMcpMatonPivotCols(int* devRrefA, int rowsA, int colsA, int rank);
std::pair<uint32_t*, uint32_t*> devGenerateProductGroup(uint32_t* devBasisPx, uint32_t* devBasisPz, int rank);
std::string decimalPairToPauliString(uint32_t Px, uint32_t Pz, int n);
int checkOdd(uint32_t Px, uint32_t Pz, int n);
void printMatOnDev(int* devA, int rows, int cols);
int getRowSum(int* devA, int rowsA, int colsA, int rowIdx, int* summandIndices, int nSummands);




class MCP {
    private:

        std::vector<std::string> pauliStrings;
        int cardinality;
        int numberOfQubits;
        int* devF2Mat = nullptr;
        int rowsF2Mat;
        int colsF2Mat; // = orderDevGamma
        int* devRrefF2Mat = nullptr;
        int rankF2Mat;
        int* devGamma = nullptr;
        int orderDevGamma;
        // int* devDecimalArray = nullptr;
        int* devF2MatPivotColIdxs = nullptr;
        uint32_t* devDecimalPx = nullptr;
        uint32_t* devDecimalPz = nullptr;

        uint32_t* devProductGroupPx = nullptr;
        uint32_t* devProductGroupPz = nullptr;

    public:

        MCP(std::vector<std::string> Pauli, int numberOfQubits) : numberOfQubits(numberOfQubits){
            this->pauliStrings = Pauli;
            this->cardinality = this->pauliStrings.size();
        };

        void printPauliStrings();
        int getCardinality();
        void generateF2Mat();

        void printF2Mat();

        void reduceF2Mat();

        void printF2MatRank();

        void printRrefF2Mat();

        void generateGamma();

        void printGamma();

        void printRrefGamma();

        void printGammaRank();

        int GammaRank();

        void findLinIndepPaulis();

        // To find bracket-dependent Paulis - find their expression as a sum in the basis
        // Then check if the sum can be written as an adjoint sequence

        void printLinIndepPaulis();

        void generateDecimalPairs();

        void printDecimalPairs();

        void removeDependentPaulis();

        int getBracketClosureSize();

        int checkBracketIndep();

        int checkBracketIndep_opt();

        std::vector<int> checkBracketIndep_vec();
        std::vector<std::string> getPaulis();
        std::vector<int> getlindepPaulis();

        void generateProductGroup();

        void printProductGroup();

        void printBracketClosure();


        ~MCP() {        // Destructor

            if (this->devF2Mat != nullptr) {
                cudaFree(devF2Mat);
            }


            if (this->devRrefF2Mat != nullptr) {
                cudaFree(devRrefF2Mat);
            }

            if (this->devGamma != nullptr) {
                cudaFree(devGamma);
            }

            // if (this->devDecimalArray != nullptr) {
            //     cudaFree(devDecimalArray);
            // }

            if (this->devF2MatPivotColIdxs != nullptr) {
                cudaFree(devF2MatPivotColIdxs);
            }

            if (this->devDecimalPx != nullptr) {
                cudaFree(devDecimalPx);
            }

            if (this->devDecimalPz != nullptr) {
                cudaFree(devDecimalPz);
            }

            if (this->devProductGroupPx != nullptr) {
                cudaFree(devProductGroupPx);
            }

            if (this->devProductGroupPz != nullptr) {
                cudaFree(devProductGroupPz);
            }

        };


};




// int main(int argc, char *argv[]) {

//     char mcpPath[2048];
//     int n;

// // #if DEFAULT_PARAMS
// //     strcpy(mcpPath, "./mcp.txt");
// //     n = 4;
// // #endif

// //     for (int i = 0; i < argc; ++i) {        // CLI Parsing
// //         if (argv[i][0] == '-') {
// //             switch(argv[i][1]) {
// //                 default:
// //                     std::cout << "Unkown option " << argv[i][1] << std::endl;
// //                     break;
// //                 case 'i': // -i ./"path_to_mcp.txt"
// //                     ++i;
// //                     strcpy(mcpPath, argv[i]);
// //                     break;
// //                 case 'n': // -i ./"path_to_mcp.txt"
// //                     ++i;
// //                     n = std::stoi(argv[i]);
// //                     break;
// //             }
// //         }
// //     }


// //     std::cout << "--- Proceeding with following parameters ---" << std::endl;
// //     std::cout << "MCP path: " << mcpPath << std::endl;
// //     std::cout << "Number of qubits: " << n << std::endl;
// //     std::cout << "--------------------------------------------" << std::endl;

// //     Pauli = ["XYZZ"]
// //     n = 4
// //     MCP mcp = MCP(Pauli, n);
// //     mcp.printPauliStrings();
// //     mcp.generateF2Mat();
// //     mcp.printF2Mat();
// //     mcp.reduceF2Mat();
// //     mcp.printF2MatRank();
// //     mcp.generateGamma();
// //     mcp.printGamma();
// //     mcp.printGammaRank();
// //     // mcp.generateBinaryArray();
// //     // mcp.printBinaryArray();
// //     mcp.generateDecimalPairs();
// //     mcp.printDecimalPairs();
// //     mcp.printRrefF2Mat();
// //     mcp.findLinIndepPaulis();
// //     mcp.printLinIndepPaulis();

// // #if 0

// //     // Remove dep stuff

// //     mcp.removeDependentPaulis();

// //     mcp.printPauliStrings();

// //     // mcp.generateF2Mat();
// //     mcp.printF2Mat();
// //     mcp.reduceF2Mat();
// //     mcp.printF2MatRank();
// //     mcp.generateGamma();
// //     mcp.printGamma();
// //     mcp.printGammaRank();
// //     // mcp.generateBinaryArray();
// //     // mcp.printBinaryArray();
// //     mcp.generateDecimalPairs();
// //     mcp.printDecimalPairs();
// //     mcp.printRrefF2Mat();
// //     mcp.findLinIndepPaulis();
// //     mcp.printLinIndepPaulis();

// //     mcp.generateProductGroup();
// //     mcp.printProductGroup();
// //     mcp.printBracketClosure();

// //     mcp.printRrefGamma();
// // #endif


// //     mcp.checkBracketIndep();


// }

