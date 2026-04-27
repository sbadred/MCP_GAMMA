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
#include <set>
#include <utility>

#include <cuda.h>
#include <cuda_runtime.h>
#include "CheckGamma.h"
// #include<cublas_v2.h>



#pragma region "CUDA checks"
// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                          \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// cusolver API error checking
// #define CUSOLVER_CHECK(err)                                                                        \
//     do {                                                                                           \
//         cusolverStatus_t err_ = (err);                                                             \
//         if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
//             printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
//             throw std::runtime_error("cusolver error");                                            \
//         }                                                                                          \
//     } while (0)

// cublas API error checking
// #define CUBLAS_CHECK(err)                                                                          \
//     do {                                                                                           \
//         cublasStatus_t err_ = (err);                                                               \
//         if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
//             printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                        \
//             throw std::runtime_error("cublas error");                                              \
//         }                                                                                          \
//     } while (0)

#pragma endregion

#define DEFAULT_PARAMS false
#define PRINT false

std::vector<std::string> readMcpTxt(const char* filePath) {

    std::ifstream ifile;
    ifile.open(filePath);

    std::string lineBuffer;
    std::vector<std::string> MCP;

    while(std::getline(ifile, lineBuffer)) {

        if (!lineBuffer.empty()) {
            MCP.push_back(lineBuffer);
        }

    }

   return MCP;

}


int* phi(std::string pauliString, int n) {

    int* f2Vec = new int[2*n];

    for (int i = 0; i < n; ++i) {

        char pauli = pauliString[i];

        switch(pauli) {
            case 'X':
                f2Vec[i] = 1;
                f2Vec[i + n] = 0;
                break;
            case 'Y':
                f2Vec[i] = 1;
                f2Vec[i + n] = 1;
                break;
            case 'Z':
                f2Vec[i] = 0;
                f2Vec[i + n] = 1;
                break;
            case 'I':
                f2Vec[i] = 0;
                f2Vec[i + n] = 0;
            default:
                break;
        }
    }

    return f2Vec;

}

bool stringOdd(const std::string& a, int QubitNumber) {
    int result = -1;
    for (int i = 0; i < QubitNumber; ++i) {
        if (a[i] == 'Y') {
            result *= -1;
        }
    }
    return result == 1;
}

bool stringEvenFlips(const std::string& a, int QubitNumber) {
    std::set<std::pair<int, int>> s;
    s.insert({0, 0});

    for (int i = 0; i < QubitNumber; ++i) {
        if (a[i] == 'X' || a[i] == 'Y') {
            std::set<std::pair<int, int>> tmp;
            if (i % 2 == 0) { // Even index
                for (const auto& element : s) {
                    tmp.insert({element.first + 1, element.second + 1});
                    tmp.insert({element.first - 1, element.second - 1});
                }
            } else { // Odd index
                for (const auto& element : s) {
                    tmp.insert({element.first + 1, element.second - 1});
                    tmp.insert({element.first - 1, element.second + 1});
                }
            }
            s = tmp;
        }
    }

    return s.find({0, 0}) != s.end();
}


std::string phiInverse(int* f2Rep, int order) {

    int numberOfQubits = (int) order/2;
    std::string pauliString;

    for(int i = 0; i < numberOfQubits; ++i) {

        int currPx = f2Rep[i];
        int currPz = f2Rep[i+numberOfQubits] + 10;

        int sum = currPx + currPz;


        switch(sum) {
            case 0:
                pauliString.push_back('I');
                break;
            case 1:
                pauliString.push_back('X');
                break;
            case 10:
                pauliString.push_back('Z');
                break;
            case 11:
                pauliString.push_back('Y');
                break;
        }
    }

    return pauliString;

}



int* hostMcpVecToDevMcpMat(std::vector<std::string> mcpVec, int n) {

    int f2VecLength = 2*n;

    int* devF2Matrix;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devF2Matrix), sizeof(int) * mcpVec.size() * f2VecLength));

    int processedElems = 0;

    for(int i = 0; i < mcpVec.size(); ++i) {

        int* f2Rep = phi(mcpVec[i], n);

        CUDA_CHECK(cudaMemcpy(&devF2Matrix[processedElems], f2Rep, sizeof(int) * f2VecLength, cudaMemcpyHostToDevice));

        delete[] f2Rep;

        processedElems += f2VecLength;

    }


    return devF2Matrix;
}








__global__ void f2DevMatmulKernel(int* devA, int* devB, int* devC, int rowsA, int colsA, int rowsB, int colsB) { // One dot prouct per thread

    int rowIdx = blockIdx.x;
    int colIdx = threadIdx.x;

    if (rowIdx < rowsA && colIdx < colsB) {
        int result = 0;
        for (int i = 0; i < colsA; ++i) {
            int a = devA[rowsA * i + rowIdx];  // A[rowIdx][i]
            int b = devB[rowsB * colIdx + i];  // B[i][colIdx]
            result ^= (a & b); // F2 addition and multiplication
        }
        devC[rowsA * colIdx + rowIdx] = result; // C[rowIdx][colIdx] - column major
    }

}


int* f2DevMatmul(int* devA, int* devB, int rowsA, int colsA, int rowsB, int colsB) {

    if(colsA != rowsB) {
        std::cout << "Incompatible matrices" << std::endl;
        return nullptr;
    }

    int rowsC = rowsA;
    int colsC = colsB;

    int* devC;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devC), sizeof(int) * rowsC * colsC));

    f2DevMatmulKernel<<<rowsC, colsC>>>(devA, devB, devC, rowsA, colsA, rowsB, colsB);

    return devC;

}
__global__ void swapRowsKernel(int* devA, int rowsA, int colsA, int swapThisRow, int withThisRow) {      // One block - one swap per thread

    int colIdx = threadIdx.x;

    if (colIdx < colsA) {
        int getToCol = colIdx * rowsA; // start index of the current column
        int temp = devA[getToCol + swapThisRow];
        devA[getToCol + swapThisRow] = devA[getToCol + withThisRow];
        devA[getToCol + withThisRow] = temp;
    }

}

__global__ void addRowsKernel(int* devA, int rowsA, int colsA, int addThisRow, int toThisRow) {      // One block - one addition per thread

    int colIdx = threadIdx.x;

    if (colIdx < colsA) {
        int getToCol = colIdx * rowsA; // start index of the current column
        devA[getToCol + toThisRow] ^= devA[getToCol + addThisRow];

        // A ^= B
        // A = A xor B
    }

}


std::pair<int*, int> f2RREF(int* devA, int rowsA, int colsA) {

    int* rrefA;
    int rank = 0;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&rrefA), sizeof(int) * rowsA * colsA));
    CUDA_CHECK(cudaMemcpy(rrefA, devA, sizeof(int) * rowsA * colsA, cudaMemcpyDeviceToDevice));

    int* hostColumn = new int[rowsA];  // alloc once at start

    ///////

    int pivotRow = 0;
    int pivotCol = 0;

    while (pivotRow < rowsA && pivotCol < colsA) {

        // 1. Find pivot: look for a row with a 1 in pivotCol starting at pivotRow

        CUDA_CHECK(cudaMemcpy(hostColumn, rrefA + pivotCol * rowsA, sizeof(int) * rowsA, cudaMemcpyDeviceToHost));

        int pivotFoundRow = -1;
        for (int r = pivotRow; r < rowsA; ++r) {
            if (hostColumn[r] == 1) {
                pivotFoundRow = r;
                break;
            }
        }

        if (pivotFoundRow == -1) {

            // No pivot in this column --> move to next column

            ++pivotCol;
            continue;
        }

        // 2. If pivot not on pivotRow --> swap rows

        if (pivotFoundRow != pivotRow) {
            swapRowsKernel<<<1, colsA>>>(rrefA, rowsA, colsA, pivotRow, pivotFoundRow);
            cudaDeviceSynchronize();
        }

        // 3. Add pivotRow to all other rows with a 1 in pivotCol

        CUDA_CHECK(cudaMemcpy(hostColumn, rrefA + pivotCol * rowsA, sizeof(int) * rowsA, cudaMemcpyDeviceToHost));

        for (int r = 0; r < rowsA; ++r) {
            if (r != pivotRow && hostColumn[r] == 1) {
                // Add pivotRow to row r
                addRowsKernel<<<1, colsA>>>(rrefA, rowsA, colsA, pivotRow, r);
                cudaDeviceSynchronize();
            }
        }

        ++pivotRow;
        ++pivotCol;
        ++rank;
    }

    delete[] hostColumn; // fixes segfault issue
    return std::pair<int*, int>(rrefA, rank);

}

int cardinalityOfProductGroup(int rank) {

    return pow(2, rank);

}




__global__ void generateGammaKernel(int* devA, int* devGamma, int rowsA, int colsA) { // One dot prouct (with \Lambda swap) per thread

    int rowIdx = blockIdx.x;
    int colIdx = threadIdx.x;

    int n = (int) (rowsA / 2);

    if (rowIdx < colsA && colIdx < colsA) {

        int result = 0;

        for (int i = 0; i < n; ++i) {
            int a = devA[rowsA * rowIdx + i];
            int b = devA[rowsA * colIdx + i + n];
            result ^= (a & b); // F2 addition and multiplication
        }

        for (int i = n; i < rowsA; ++i) {
            int a = devA[rowsA * rowIdx + i];
            int b = devA[rowsA * colIdx + i - n];
            result ^= (a & b); // F2 addition and multiplication
        }

        devGamma[colsA * colIdx + rowIdx] = result; // devGamma[rowIdx][colIdx] - column major

    }

}


int* devGenerateGamma(int* devA, int rowsA, int colsA) {

    int* devGamma;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devGamma), sizeof(int) * colsA * colsA));

    generateGammaKernel<<<colsA, colsA>>>(devA, devGamma, rowsA, colsA);

    return devGamma;

}


__global__ void devF2MatToDecimalPairsKernel(int* devA, int rowsA, int colsA, uint32_t* devPx, uint32_t* devPz) {    // one block - one thread per column

    int colIdx = threadIdx.x;

    if (colIdx < colsA) {

        int n = (int)rowsA /2;
        // printf("%d", n);
        uint32_t pX = 0;
        uint32_t pZ = 0;
        uint32_t base = 1;
        uint32_t getToCol = rowsA * colIdx;

        for(int i = 0; i < n; ++i) {
            pX += devA[getToCol + (n-1) - i] * base;
            base *= 2;
        }

        base = 1;

        for(int i = 0; i < n; ++i) {
            pZ += devA[getToCol + (rowsA-1) - i] * base;
            base *= 2;
        }

        devPx[colIdx] = pX;
        devPz[colIdx] = pZ;
    }
}

std::pair<uint32_t*, uint32_t*> devF2MatToDecimalPairs(int* devA, int rowsA, int colsA) {

    uint32_t* devDecimalPx;
    uint32_t* devDecimalPz;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devDecimalPx), sizeof(uint32_t) * colsA));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devDecimalPz), sizeof(uint32_t) * colsA));


    devF2MatToDecimalPairsKernel<<<1, colsA>>>(devA, rowsA, colsA, devDecimalPx, devDecimalPz);

    return std::make_pair(devDecimalPx, devDecimalPz);

}

__global__ void devFindPivotColsKernel(int* devRrefA, int rowsA, int colsA, int* pivotColIdx, int rank) {       // one block, one thread

    int nPivotCols = 0;

    for (int col = 0; col < colsA; ++col) {
        if (devRrefA[col * rowsA + nPivotCols] == 1) {
            pivotColIdx[nPivotCols] = col;
            ++nPivotCols;
        }
    }

}


int* devFindPivotCols(int* devRrefA, int rowsA, int colsA, int rank) {

    int* pivotColIdxs;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pivotColIdxs), sizeof(int) * rank));

    devFindPivotColsKernel<<<1,1>>>(devRrefA, rowsA, colsA, pivotColIdxs, rank);

    return pivotColIdxs;

}

__global__ void devFindPivotAndNonPivotColsKernel(int* devRrefA, int rowsA, int colsA, int* pivotColIdx, int* nonPivotColsIdx, int rank) {       // one block, one thread

    int nPivotCols = 0;
    int nNonPivotCols = 0;

    for (int col = 0; col < colsA; ++col) {
        if (devRrefA[col * rowsA + nPivotCols] == 1) {
            pivotColIdx[nPivotCols] = col;
            ++nPivotCols;
        } else {
            nonPivotColsIdx[nNonPivotCols] = col;
            ++nNonPivotCols;
        }
    }

}



std::pair<int*, int*> devFindPivotAndNonPivotCols(int* devRrefA, int rowsA, int colsA, int rank) {

    int* pivotColIdxs;
    int* nonPivotColsIdxs;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pivotColIdxs), sizeof(int) * rank));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&nonPivotColsIdxs), sizeof(int) * (colsA - rank)));


    devFindPivotAndNonPivotColsKernel<<<1,1>>>(devRrefA, rowsA, colsA, pivotColIdxs, nonPivotColsIdxs, rank);

    return std::make_pair(pivotColIdxs, nonPivotColsIdxs);

}


__global__ void xorKernel(uint32_t* devBasisPx, uint32_t* devBasisPz, int rank, uint32_t* devProductGroupPx, uint32_t* devProductGroupPz, uint32_t cardProductGroup) {    // One xor sum per thread - idx goes from 0000000... to 11111111... (i.e. 0 to 2^r - 1)

    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // idx decomposed in binary gives corresponding unique sum


    if (idx < cardProductGroup) {

        uint32_t xorPx = 0;
        uint32_t xorPz = 0;


        // Do xor sum -- one per thread
        for (int i = 0; i < rank; ++i) {
            if ((idx >> i) & 1) {  // Check ith bit of idx
                xorPx ^= devBasisPx[i];  // XOR
                xorPz ^= devBasisPz[i];
            }
        }

        devProductGroupPx[idx] = xorPx;
        devProductGroupPz[idx] = xorPz;
    }
}


std::pair<uint32_t*, uint32_t*> devGenerateProductGroup(uint32_t* devBasisPx, uint32_t* devBasisPz, int rank) {

    uint32_t cardProductGroup = 1 << rank;  // 2^rank

    uint32_t* devProductGroupPx;
    uint32_t* devProductGroupPz;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devProductGroupPx), sizeof(uint32_t) * cardProductGroup));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devProductGroupPz), sizeof(uint32_t) * cardProductGroup));

    int numberOfBlocks = int (cardProductGroup + 255 / 256);
    xorKernel<<<numberOfBlocks, 256 >>>(devBasisPx, devBasisPz, rank, devProductGroupPx, devProductGroupPz, cardProductGroup);


    return std::make_pair(devProductGroupPx, devProductGroupPz);

}



std::string decimalPairToPauliString(uint32_t Px, uint32_t Pz, int n) {  // where Px and Pz are both of length n

    std::string pauliString;

    for(int i = 0; i < n; ++i) {

        uint32_t currPx = (Px >> i) & 1;  // Right shift and mask to get bit at position i
        uint32_t currPz = (Pz >> i) & 1;  // Right shift and mask to get bit at position i

        // std::cout << currPx << " " << currPz << std::endl;

        uint32_t sum = currPx + currPz * 10;


        switch(sum) {
            case 0:
                // pauliString.push_back('I');
                pauliString.insert(pauliString.begin(), 'I');
                break;
            case 1:
                // pauliString.push_back('X');
                pauliString.insert(pauliString.begin(), 'X');

                break;
            case 10:
                // pauliString.push_back('Z');
                pauliString.insert(pauliString.begin(), 'Z');

                break;
            case 11:
                // pauliString.push_back('Y');
                pauliString.insert(pauliString.begin(), 'Y');

                break;
        }
    }

    return pauliString;

}



int checkOdd(uint32_t Px, uint32_t Pz, int n) { // n = number of qubits
    uint32_t res = Px & Pz;

    // std::bitset<32> b(res); // 32 bits for uint32_t
    // std::cout << b << std::endl;

    int count = 0;

    for (int i = 0; i < n; ++i) {
        if((res >> i) & 1) {
            ++count;
        }
    }
    // std::cout << count << std::endl;
    return count % 2;
}



void printMatOnDev(int* devA, int rows, int cols) {



    int* hostF2Mat = new int[rows * cols];
    CUDA_CHECK(cudaMemcpy(hostF2Mat, devA, sizeof(int) * rows * cols, cudaMemcpyDeviceToHost));
    std::cout << "--- Matrix ---" << std::endl;
    for (int r = 0; r < rows; ++r) {
        // std::cout << "Row " << r << ": ";
        for (int c = 0; c < cols; ++c) {
            std::cout << hostF2Mat[c * rows + r] << " ";

        }
        std::cout << std::endl;
    }
    std::cout << std::endl;


    delete[] hostF2Mat;

};



__global__ void getRowSumKernel(int* devA, int rowsA, int colsA, int rowIdx, int* devSummandIndices, int nSummands, int* sum) {

    *sum = 0;

    for(int i = 0; i < nSummands; ++i) {
        // printf("test");
        // printf("%d\n", devA[rowsA * devSummandIndices[i] + rowIdx]);


        *sum ^= devA[rowsA * devSummandIndices[i] + rowIdx];
    }
}



int getRowSum(int* devA, int rowsA, int colsA, int rowIdx, int* summandIndices, int nSummands) {

    int* sum;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sum), sizeof(int)));

    int* devSummandIndices;
    cudaMalloc(reinterpret_cast<void**>(&devSummandIndices), sizeof(int) * nSummands);
    cudaMemcpy(devSummandIndices, summandIndices, sizeof(int) * nSummands, cudaMemcpyHostToDevice);

    // for (int i = 0; i < nSummands; ++i) {
    //     std::cout << "TEST " << summandIndices[i] << std::endl;
    // }

    getRowSumKernel<<<1,1>>>(devA, rowsA, colsA, rowIdx, devSummandIndices, nSummands, sum);

    cudaDeviceSynchronize();

    int res;
    CUDA_CHECK(cudaMemcpy(&res, sum, sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(sum);
    cudaFree(devSummandIndices);

    return res;


}





void MCP::generateF2Mat() {
            this->devF2Mat = hostMcpVecToDevMcpMat(this->pauliStrings, this->numberOfQubits);
            this->rowsF2Mat = 2 * this->numberOfQubits;
            this->colsF2Mat = this->cardinality;

            //std::cout << rowsF2Mat << numberOfQubits <<  " " << this->colsF2Mat << std::endl;
        };

void MCP::printF2Mat() {

    int* hostF2Mat = new int[this->rowsF2Mat * this->colsF2Mat];
    CUDA_CHECK(cudaMemcpy(hostF2Mat, this->devF2Mat, sizeof(int) * this->rowsF2Mat * this->colsF2Mat, cudaMemcpyDeviceToHost));

    std::cout << "--- phi(A) Matrix ---" << std::endl;
    std::cout << "dimensions: " <<  this->rowsF2Mat << " "<< this->colsF2Mat << std::endl;

    for (int r = 0; r < this->rowsF2Mat; ++r) {
        // std::cout << "Row " << r << ": ";
        for (int c = 0; c < this->colsF2Mat; ++c) {
            std::cout << hostF2Mat[c * this->rowsF2Mat + r] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    delete[] hostF2Mat;

};

void MCP::reduceF2Mat() {
            std::pair<int*, int> temp = f2RREF(this->devF2Mat, this->rowsF2Mat, this->colsF2Mat);
            this->devRrefF2Mat = temp.first;
            this->rankF2Mat = temp.second;
        };

void MCP::printF2MatRank() {
            std::cout << "Rank of phi(A): " << this->rankF2Mat << std::endl;
            std::cout << std::endl;
        };

void MCP::printRrefF2Mat() {

            int* hostF2Mat = new int[this->rowsF2Mat * this->colsF2Mat];
            CUDA_CHECK(cudaMemcpy(hostF2Mat, this->devRrefF2Mat, sizeof(int) * this->rowsF2Mat * this->colsF2Mat, cudaMemcpyDeviceToHost));

            std::cout << "--- RREF of phi(A) Matrix ---" << std::endl;

            for (int r = 0; r < this->rowsF2Mat; ++r) {
                // std::cout << "Row " << r << ": ";
                for (int c = 0; c < this->colsF2Mat; ++c) {
                    std::cout << hostF2Mat[c * this->rowsF2Mat + r] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;

            delete[] hostF2Mat;

        };

void MCP::generateGamma() {

            this->devGamma = devGenerateGamma(this->devF2Mat, this->rowsF2Mat, this->colsF2Mat);
            this->orderDevGamma = this->colsF2Mat;

        };

void MCP::printGamma() {

            int* hostGamma = new int[this->orderDevGamma * this->orderDevGamma];
            CUDA_CHECK(cudaMemcpy(hostGamma, this->devGamma, sizeof(int) * this->orderDevGamma * this->orderDevGamma, cudaMemcpyDeviceToHost));

            std::cout << "--- Gamma Matrix ---" << std::endl;

            for (int r = 0; r < this->orderDevGamma; ++r) {
                // std::cout << "Row " << r << ": ";
                for (int c = 0; c < this->orderDevGamma; ++c) {
                    std::cout << hostGamma[c * this->orderDevGamma + r] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;

            delete[] hostGamma;

        };

void MCP::printRrefGamma() {     // I'm not bothering adding the RrefGamma as a class memeber. I might do it eventually if it's useful

            std::pair<int*, int> temp = f2RREF(this->devGamma, this->orderDevGamma, this->orderDevGamma);

            int* hostGamma = new int[this->orderDevGamma * this->orderDevGamma];
            CUDA_CHECK(cudaMemcpy(hostGamma, temp.first, sizeof(int) * this->orderDevGamma * this->orderDevGamma, cudaMemcpyDeviceToHost));

            std::cout << "--- RREF of Gamma Matrix ---" << std::endl;

            for (int r = 0; r < this->orderDevGamma; ++r) {
                // std::cout << "Row " << r << ": ";
                for (int c = 0; c < this->orderDevGamma; ++c) {
                    std::cout << hostGamma[c * this->orderDevGamma + r] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;

            delete[] hostGamma;

            cudaFree(temp.first);

        }

void MCP::printGammaRank() {
            std::pair<int*, int> temp = f2RREF(this->devGamma, this->orderDevGamma, this->orderDevGamma);
            std::cout << "Rank of Gamma: " << temp.second << std::endl;
            cudaFree(temp.first);
        }


int MCP::GammaRank() {
            std::pair<int*, int> temp = f2RREF(this->devGamma, this->orderDevGamma, this->orderDevGamma);
            std::cout << "Rank of Gamma: " << temp.second << std::endl;
            cudaFree(temp.first);
            return temp.second;
        }

void MCP::findLinIndepPaulis() {     // Can be used to find bracket-dependent Paulis
            this->devF2MatPivotColIdxs = devFindPivotCols(this->devRrefF2Mat, this->rowsF2Mat, this->colsF2Mat, this->rankF2Mat);
        }



void MCP::printLinIndepPaulis() {

            int* pivotColIdxs = new int[this->rankF2Mat];
            CUDA_CHECK(cudaMemcpy(pivotColIdxs, this->devF2MatPivotColIdxs, sizeof(int) * this->rankF2Mat, cudaMemcpyDeviceToHost));

            std::cout << "Linearly Independent Paulis in Set" << std::endl;

            for (int i = 0; i < this->rankF2Mat; ++i) {
                std::cout << pivotColIdxs[i] << " ";
            }
            std::cout << std::endl;
        }



void MCP::generateDecimalPairs() {
            std::pair<uint32_t*, uint32_t*> temp = devF2MatToDecimalPairs(this->devF2Mat, rowsF2Mat, colsF2Mat);
            this->devDecimalPx = temp.first;
            this->devDecimalPz = temp.second;
        }

void MCP::printDecimalPairs() {

            int* hostPx = new int[this->colsF2Mat];
            int* hostPz = new int[this->colsF2Mat];

            CUDA_CHECK(cudaMemcpy(hostPx, this->devDecimalPx, sizeof(int) * this->colsF2Mat, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hostPz, this->devDecimalPz, sizeof(int) * this->colsF2Mat, cudaMemcpyDeviceToHost));


            std::cout << "---Decimal Pairs---" << std::endl;

            for(int i = 0; i < colsF2Mat; ++i) {
                std::cout <<  "(" << hostPx[i] << ", " << hostPz[i] << ")" << " ";
            }
            std::cout << std::endl;

            delete[] hostPx;
            delete[] hostPz;
        }


void MCP::removeDependentPaulis() {      // Removes product-dependent Paulis
            // Needs this->devF2MatPivotColIdxs to be generated
            if (this->rankF2Mat != min(rowsF2Mat, colsF2Mat)) {

                int* newF2Mat;
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&newF2Mat), sizeof(int) * this->rowsF2Mat * this->rankF2Mat));
                int processedElems = 0;

                std::vector<std::string> newPauliStrings;

                int* hostPivotCols = new int[this->rankF2Mat];
                CUDA_CHECK(cudaMemcpy(hostPivotCols, this->devF2MatPivotColIdxs, sizeof(int) * this->rankF2Mat, cudaMemcpyDeviceToHost));

                for(int i = 0; i < this->rankF2Mat; ++i) {

                    // std::cout << "i: " << i << std::endl;

                    int colNumber = hostPivotCols[i];

                    // std::cout << colNumber << std::endl;

                    CUDA_CHECK(cudaMemcpy(&newF2Mat[processedElems], &devF2Mat[rowsF2Mat * colNumber], sizeof(int) * rowsF2Mat, cudaMemcpyDeviceToDevice));

                    processedElems += this->rowsF2Mat;

                    newPauliStrings.push_back(this->pauliStrings[colNumber]);
                }

                delete[] hostPivotCols;

                CUDA_CHECK(cudaFree(this->devF2Mat));
                this->devF2Mat = newF2Mat;

                this->colsF2Mat = this->rankF2Mat;
                this->cardinality = this->rankF2Mat;

                this->pauliStrings.clear();
                this->pauliStrings = newPauliStrings;


                if (this->devRrefF2Mat != nullptr) {
                    cudaFree(devRrefF2Mat);
                }

                if (this->devDecimalPx != nullptr) {
                    cudaFree(devDecimalPx);
                }

                if (this->devDecimalPz != nullptr) {
                    cudaFree(devDecimalPx);
                }

                if (this->devDecimalPz != nullptr) {
                    cudaFree(devDecimalPx);
                }

                if (this->devF2MatPivotColIdxs != nullptr) {
                    cudaFree(devF2MatPivotColIdxs);
                }

                if (this->devGamma != nullptr) {
                    cudaFree(devGamma);
                }


            }
        }


std::vector<int> MCP::getlindepPaulis()  {
    std::vector<int> lindep;
    int* hostNonPivotCols = new int[this->colsF2Mat - this->rankF2Mat];
    std::pair<int*, int*> temp = devFindPivotAndNonPivotCols(this->devRrefF2Mat, this->rowsF2Mat, this->colsF2Mat, this->rankF2Mat);
    cudaFree(temp.first);
    cudaMemcpy(hostNonPivotCols, temp.second, sizeof(int) * (this->colsF2Mat - this->rankF2Mat), cudaMemcpyDeviceToHost);
    cudaFree(temp.second);

    for(int i = 0; i < (this->colsF2Mat - this->rankF2Mat); ++i) {
            int colNumber = hostNonPivotCols[i];

            std::cout << "Pauli at index " << colNumber << " in the set is linearly-dependent "<< std::endl;
            lindep.push_back(colNumber);

    }
    return lindep;

}


int MCP::checkBracketIndep() {
            // std::cout << rankF2Mat  << std::endl;
            if (this->rankF2Mat < min(rowsF2Mat, colsF2Mat)) {

                int* newF2Mat;
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&newF2Mat), sizeof(int) * this->rowsF2Mat * this->colsF2Mat));   // Copy L.I. vectors first, then L.D. vecs
                int processedElems = 0;

                std::vector<std::string> newPauliStrings;

                int* hostPivotCols = new int[this->rankF2Mat];

                std::pair<int*, int*> temp = devFindPivotAndNonPivotCols(this->devRrefF2Mat, this->rowsF2Mat, this->colsF2Mat, this->rankF2Mat);

                CUDA_CHECK(cudaMemcpy(hostPivotCols, temp.first, sizeof(int) * this->rankF2Mat, cudaMemcpyDeviceToHost));

                cudaFree(temp.first);

                for(int i = 0; i < this->rankF2Mat; ++i) {

                    // std::cout << "i: " << i << std::endl;

                    int colNumber = hostPivotCols[i];

                    // std::cout << colNumber << std::endl;

                    CUDA_CHECK(cudaMemcpy(&newF2Mat[processedElems], &devF2Mat[rowsF2Mat * colNumber], sizeof(int) * rowsF2Mat, cudaMemcpyDeviceToDevice));

                    processedElems += this->rowsF2Mat;

                    newPauliStrings.push_back(this->pauliStrings[colNumber]);
                }

                delete[] hostPivotCols;

                int* hostNonPivotCols = new int[this->colsF2Mat - this->rankF2Mat];

                cudaMemcpy(hostNonPivotCols, temp.second, sizeof(int) * (this->colsF2Mat - this->rankF2Mat), cudaMemcpyDeviceToHost);
                cudaFree(temp.second);


                for(int i = 0; i < (this->colsF2Mat - this->rankF2Mat); ++i) {

                    // std::cout << "i: " << i << std::endl;

                    int colNumber = hostNonPivotCols[i];

                    std::cout << "Pauli at index " << colNumber << " in the set is linearly-dependent "<< std::endl;

                    CUDA_CHECK(cudaMemcpy(&newF2Mat[processedElems], &devF2Mat[rowsF2Mat * colNumber], sizeof(int) * rowsF2Mat, cudaMemcpyDeviceToDevice));

                    processedElems += this->rowsF2Mat;

                }       // Don't delte nonPivotCols just yet

                // APPLY RREF TO AUGMENTED MATRIX TO GET EXPR OF DEPENDENT COLS (copied to the back)

                std::pair<int*, int> augmentedRref = f2RREF(newF2Mat, this->rowsF2Mat, this->colsF2Mat);
                int* reducedNewMat = augmentedRref.first;

#if 0
                printMatOnDev(newF2Mat, this->rowsF2Mat, this->colsF2Mat);
#endif
                cudaFree(newF2Mat);

#if 0
                printMatOnDev(reducedNewMat, this->rowsF2Mat, this->colsF2Mat);     // Last cols - rank columns of this contain expression of nonPivotCols
#endif


                bool bracketIndep = true;

                for(int i = 0; i < (this->colsF2Mat - this->rankF2Mat); ++i) {

                    int* currRep = new int[this->rowsF2Mat];
                    cudaMemcpy(currRep, &reducedNewMat[this->rankF2Mat * this->rowsF2Mat + i * this->rowsF2Mat], sizeof(int) * this->rowsF2Mat, cudaMemcpyDeviceToHost);

                    std::cout << "alpha_" << hostNonPivotCols[i] << " is equal to: "<< std::endl;

                    bool first = true;
                    std::vector<int> summandIndices;

                    for(int j = 0; j < this->rowsF2Mat; ++j) {
                        // int nSummands = 0; -- just get from summands.size()
                        // std::cout << currRep[j] << std::endl;
                        if(currRep[j] == 1) {
                            // ++nSummands;
                            summandIndices.push_back(j);
                            if (first) {
                                std::cout << "alpha_" << j;
                                first = false;
                            } else {
                                std::cout << " + "<< "alpha_" << j;
                            }
                        }

                        // Now we need to check every possible sum order. -- one thread, one sum
                        // if (summandIndices.size() > 1) {
                        //     bool currBrackDep = checkSums(this->devGamma, summandIndices);
                        // } --- scrapped idea
                    }

                    delete[] currRep;

                    std::cout << std::endl;

                    if (summandIndices.size() == 1) {
                        std::cout << "Set is Bracket-Dependent because this element is equal to another one in the set." << std::endl;
                        // bracketIndep = false;
                        continue;
                    }

#if 0
                    // DEAL WITH SUM HERE
                    std::vector<int> workingSummands(summandIndices);

                    for (auto summand : summandIndices) {

                        // std::cout << "Summand " << summand << std::endl;

                        bool keepGoing = true;
                        int sumSize = 0;

                        if (getRowSum(this->devGamma, this->orderDevGamma, this->orderDevGamma, summand, summandIndices.data(), summandIndices.size())) {
                            ++sumSize;
                            workingSummands.erase(
                                std::remove(workingSummands.begin(), workingSummands.end(), summand),
                                workingSummands.end()
                            );

                            // while(keepGoing) {
                            //     for(auto s : workingSummands) {

                            //         if(getRowSum(this->devGamma, this->orderDevGamma, this->orderDevGamma, s, workingSummands.data(), workingSummands.size())) {
                            //             ++sumSize;
                            //             workingSummands.erase(
                            //                 std::remove(workingSummands.begin(), workingSummands.end(), s),
                            //                 workingSummands.end()
                            //             );

                            //         }

                            //     }

                            // }
                            // std::cout << "TEST" << std::endl;

                        }


                        // std::cout << "Row sum " << getRowSum(this->devGamma, this->orderDevGamma, this->orderDevGamma, summand, summandIndices.data(), summandIndices.size()) << std::endl;

                    }

                    // bracketDep = false;

#endif
                    // For now, just do all perms -- at least they have an easy exit cond.
                    bool foundValidAdjointSequence= false;
                    // std::sort(summandIndices.begin(), summandIndices.end()); -- already sorted -- not needed

                    do {
                        std::vector<int> permutation(summandIndices); // A specific permutation
                        bool validSequence = true;

                        for (size_t step = 0; step < permutation.size() - 1; ++step) {

                            int currentSummand = permutation[step];

                            // Build the set of remaining summands (step+1 onward)
                            std::vector<int> remaining(permutation.begin() + step + 1, permutation.end());

                            if (remaining.empty()) {
                                validSequence = false;
                                break; // No one left to sum over
                            }

                            if (!getRowSum(this->devGamma, this->orderDevGamma, this->orderDevGamma,
                                           currentSummand, remaining.data(), static_cast<int>(remaining.size()))) {
                                validSequence = false;
                                break; // This permutation doesn't work
                            }
                        }

                        // Last summand (leaf node): nothing left to sum with, so it's invalid
                        // could allow it if there's no sum required anymore, tho
                        // final element can't sum over empty set, tho ??
                        // --> assume last element is a leaf and not allowed

                        if (validSequence) {
                            foundValidAdjointSequence = true;
                            std::cout << "Valid adjoint sequence found for alpha_" << hostNonPivotCols[i] << ": ";
                            for (auto idx : permutation) {
                                std::cout << "alpha_" << idx << " ";
                            }
                            std::cout << std::endl;
                            break;
                        }


                    } while (std::next_permutation(summandIndices.begin(), summandIndices.end()));

                    if (!foundValidAdjointSequence){
                        return 0;}


                }






            }
            return 1;

        }


int MCP::checkBracketIndep_opt() {
            // std::cout << rankF2Mat  << std::endl;
            int value=1;

            if (this->rankF2Mat < min(rowsF2Mat, colsF2Mat)) {

                int* newF2Mat;
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&newF2Mat), sizeof(int) * this->rowsF2Mat * this->colsF2Mat));   // Copy L.I. vectors first, then L.D. vecs
                int processedElems = 0;

                std::vector<std::string> newPauliStrings;

                int* hostPivotCols = new int[this->rankF2Mat];

                std::pair<int*, int*> temp = devFindPivotAndNonPivotCols(this->devRrefF2Mat, this->rowsF2Mat, this->colsF2Mat, this->rankF2Mat);

                CUDA_CHECK(cudaMemcpy(hostPivotCols, temp.first, sizeof(int) * this->rankF2Mat, cudaMemcpyDeviceToHost));

                cudaFree(temp.first);

                for(int i = 0; i < this->rankF2Mat; ++i) {

                    // std::cout << "i: " << i << std::endl;

                    int colNumber = hostPivotCols[i];

                    // std::cout << colNumber << std::endl;

                    CUDA_CHECK(cudaMemcpy(&newF2Mat[processedElems], &devF2Mat[rowsF2Mat * colNumber], sizeof(int) * rowsF2Mat, cudaMemcpyDeviceToDevice));

                    processedElems += this->rowsF2Mat;

                    newPauliStrings.push_back(this->pauliStrings[colNumber]);
                }

                delete[] hostPivotCols;

                int* hostNonPivotCols = new int[this->colsF2Mat - this->rankF2Mat];

                cudaMemcpy(hostNonPivotCols, temp.second, sizeof(int) * (this->colsF2Mat - this->rankF2Mat), cudaMemcpyDeviceToHost);
                cudaFree(temp.second);


                for(int i = 0; i < (this->colsF2Mat - this->rankF2Mat); ++i) {

                    // std::cout << "i: " << i << std::endl;

                    int colNumber = hostNonPivotCols[i];

                    std::cout << "Pauli at index " << colNumber << " in the set is linearly-dependent "<< std::endl;

                    CUDA_CHECK(cudaMemcpy(&newF2Mat[processedElems], &devF2Mat[rowsF2Mat * colNumber], sizeof(int) * rowsF2Mat, cudaMemcpyDeviceToDevice));

                    processedElems += this->rowsF2Mat;

                }       // Don't delte nonPivotCols just yet

                // APPLY RREF TO AUGMENTED MATRIX TO GET EXPR OF DEPENDENT COLS (copied to the back)

                std::pair<int*, int> augmentedRref = f2RREF(newF2Mat, this->rowsF2Mat, this->colsF2Mat);
                int* reducedNewMat = augmentedRref.first;

#if 0
                printMatOnDev(newF2Mat, this->rowsF2Mat, this->colsF2Mat);
#endif
                cudaFree(newF2Mat);

#if 0
                printMatOnDev(reducedNewMat, this->rowsF2Mat, this->colsF2Mat);     // Last cols - rank columns of this contain expression of nonPivotCols
#endif


                bool bracketIndep = true;
                //for(int i = 0; i < (this->colsF2Mat - this->rankF2Mat); ++i) {

                for(int i = (this->colsF2Mat - this->rankF2Mat)-1; i < (this->colsF2Mat - this->rankF2Mat); ++i) {

                    int* currRep = new int[this->rowsF2Mat];
                    cudaMemcpy(currRep, &reducedNewMat[this->rankF2Mat * this->rowsF2Mat + i * this->rowsF2Mat], sizeof(int) * this->rowsF2Mat, cudaMemcpyDeviceToHost);

                    std::cout << "alpha_" << hostNonPivotCols[i] << " is equal to: "<< std::endl;

                    bool first = true;
                    std::vector<int> summandIndices;

                    for(int j = 0; j < this->rowsF2Mat; ++j) {
                        // int nSummands = 0; -- just get from summands.size()
                        // std::cout << currRep[j] << std::endl;
                        if(currRep[j] == 1) {
                            // ++nSummands;
                            summandIndices.push_back(j);
                            if (first) {
                                std::cout << "alpha_" << j;
                                first = false;
                            } else {
                                std::cout << " + "<< "alpha_" << j;
                            }
                        }

                        // Now we need to check every possible sum order. -- one thread, one sum
                        // if (summandIndices.size() > 1) {
                        //     bool currBrackDep = checkSums(this->devGamma, summandIndices);
                        // } --- scrapped idea
                    }

                    delete[] currRep;

                    std::cout << std::endl;

                    if (summandIndices.size() == 1) {
                        std::cout << "Set is Bracket-Dependent because this element is equal to another one in the set." << std::endl;
                        value = 0;
                        // bracketIndep = false;
                        continue;
                    }

                    if (summandIndices.size() > 7) {
                        std::cout << "Too many sums! " << std::endl;
                        value = 0;
                        // bracketIndep = false;
                        continue;
                    }

#if 0
                    // DEAL WITH SUM HERE
                    std::vector<int> workingSummands(summandIndices);

                    for (auto summand : summandIndices) {

                        // std::cout << "Summand " << summand << std::endl;

                        bool keepGoing = true;
                        int sumSize = 0;

                        if (getRowSum(this->devGamma, this->orderDevGamma, this->orderDevGamma, summand, summandIndices.data(), summandIndices.size())) {
                            ++sumSize;
                            workingSummands.erase(
                                std::remove(workingSummands.begin(), workingSummands.end(), summand),
                                workingSummands.end()
                            );

                            // while(keepGoing) {
                            //     for(auto s : workingSummands) {

                            //         if(getRowSum(this->devGamma, this->orderDevGamma, this->orderDevGamma, s, workingSummands.data(), workingSummands.size())) {
                            //             ++sumSize;
                            //             workingSummands.erase(
                            //                 std::remove(workingSummands.begin(), workingSummands.end(), s),
                            //                 workingSummands.end()
                            //             );

                            //         }

                            //     }

                            // }
                            // std::cout << "TEST" << std::endl;

                        }


                        // std::cout << "Row sum " << getRowSum(this->devGamma, this->orderDevGamma, this->orderDevGamma, summand, summandIndices.data(), summandIndices.size()) << std::endl;

                    }

                    // bracketDep = false;

#endif
                    // For now, just do all perms -- at least they have an easy exit cond.
                    bool foundValidAdjointSequence= false;
                    // std::sort(summandIndices.begin(), summandIndices.end()); -- already sorted -- not needed

                    do {
                        std::vector<int> permutation(summandIndices); // A specific permutation
                        bool validSequence = true;

                        for (size_t step = 0; step < permutation.size() - 1; ++step) {

                            int currentSummand = permutation[step];

                            // Build the set of remaining summands (step+1 onward)
                            std::vector<int> remaining(permutation.begin() + step + 1, permutation.end());

                            if (remaining.empty()) {
                                validSequence = false;
                                break; // No one left to sum over
                            }

                            if (!getRowSum(this->devGamma, this->orderDevGamma, this->orderDevGamma,
                                           currentSummand, remaining.data(), static_cast<int>(remaining.size()))) {
                                validSequence = false;
                                break; // This permutation doesn't work
                            }
                        }

                        // Last summand (leaf node): nothing left to sum with, so it's invalid
                        // could allow it if there's no sum required anymore, tho
                        // final element can't sum over empty set, tho ??
                        // --> assume last element is a leaf and not allowed

                        if (validSequence) {
                            foundValidAdjointSequence = true;
                            std::cout << "Valid adjoint sequence found for alpha_" << hostNonPivotCols[i] << ": ";
                            for (auto idx : permutation) {
                                std::cout << "alpha_" << idx << " ";
                            }
                            std::cout << std::endl;
                            value =0;
                            break;
                        }


                    } while (std::next_permutation(summandIndices.begin(), summandIndices.end()));

                    if (!foundValidAdjointSequence){
                        std::cout << "Not Valid adjoint sequence found for alpha_" << hostNonPivotCols[i] << "\n"<< std::endl;
                        return 1;}
                    else
                    {  return 0;
                    }


                }






            }
            return value;

        }


std::vector<int> MCP::checkBracketIndep_vec() {
        std::vector<int> left_vec;
        // std::cout << rankF2Mat  << std::endl;
        if (this->rankF2Mat < min(rowsF2Mat, colsF2Mat)) {

            int* newF2Mat;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&newF2Mat), sizeof(int) * this->rowsF2Mat * this->colsF2Mat));   // Copy L.I. vectors first, then L.D. vecs
            int processedElems = 0;

            std::vector<std::string> newPauliStrings;

            int* hostPivotCols = new int[this->rankF2Mat];

            std::pair<int*, int*> temp = devFindPivotAndNonPivotCols(this->devRrefF2Mat, this->rowsF2Mat, this->colsF2Mat, this->rankF2Mat);

            CUDA_CHECK(cudaMemcpy(hostPivotCols, temp.first, sizeof(int) * this->rankF2Mat, cudaMemcpyDeviceToHost));

            cudaFree(temp.first);

            for(int i = 0; i < this->rankF2Mat; ++i) {

                // std::cout << "i: " << i << std::endl;

                int colNumber = hostPivotCols[i];

                // std::cout << colNumber << std::endl;

                CUDA_CHECK(cudaMemcpy(&newF2Mat[processedElems], &devF2Mat[rowsF2Mat * colNumber], sizeof(int) * rowsF2Mat, cudaMemcpyDeviceToDevice));

                processedElems += this->rowsF2Mat;

                newPauliStrings.push_back(this->pauliStrings[colNumber]);
            }

            delete[] hostPivotCols;

            int* hostNonPivotCols = new int[this->colsF2Mat - this->rankF2Mat];

            cudaMemcpy(hostNonPivotCols, temp.second, sizeof(int) * (this->colsF2Mat - this->rankF2Mat), cudaMemcpyDeviceToHost);
            cudaFree(temp.second);


            for(int i = 0; i < (this->colsF2Mat - this->rankF2Mat); ++i) {

                // std::cout << "i: " << i << std::endl;

                int colNumber = hostNonPivotCols[i];

                std::cout << "Pauli at index " << colNumber << " in the set is linearly-dependent "<< std::endl;

                CUDA_CHECK(cudaMemcpy(&newF2Mat[processedElems], &devF2Mat[rowsF2Mat * colNumber], sizeof(int) * rowsF2Mat, cudaMemcpyDeviceToDevice));

                processedElems += this->rowsF2Mat;

            }       // Don't delte nonPivotCols just yet

            // APPLY RREF TO AUGMENTED MATRIX TO GET EXPR OF DEPENDENT COLS (copied to the back)

            std::pair<int*, int> augmentedRref = f2RREF(newF2Mat, this->rowsF2Mat, this->colsF2Mat);
            int* reducedNewMat = augmentedRref.first;

#if 0
            printMatOnDev(newF2Mat, this->rowsF2Mat, this->colsF2Mat);
#endif
            cudaFree(newF2Mat);

#if 0
            printMatOnDev(reducedNewMat, this->rowsF2Mat, this->colsF2Mat);     // Last cols - rank columns of this contain expression of nonPivotCols
#endif


            // bool bracketIndep = true;


            for(int i = 0; i < (this->colsF2Mat - this->rankF2Mat); ++i) {

                int* currRep = new int[this->rowsF2Mat];
                cudaMemcpy(currRep, &reducedNewMat[this->rankF2Mat * this->rowsF2Mat + i * this->rowsF2Mat], sizeof(int) * this->rowsF2Mat, cudaMemcpyDeviceToHost);

                std::cout << "alpha_" << hostNonPivotCols[i] << " is equal to: "<< std::endl;

                bool first = true;
                std::vector<int> summandIndices;

                for(int j = 0; j < this->rowsF2Mat; ++j) {
                    // int nSummands = 0; -- just get from summands.size()
                    // std::cout << currRep[j] << std::endl;
                    if(currRep[j] == 1) {
                        // ++nSummands;
                        summandIndices.push_back(j);
                        if (first) {
                            std::cout << "alpha_" << j;
                            first = false;
                        } else {
                            std::cout << " + "<< "alpha_" << j;
                        }
                    }

                    // Now we need to check every possible sum order. -- one thread, one sum
                    // if (summandIndices.size() > 1) {
                    //     bool currBrackDep = checkSums(this->devGamma, summandIndices);
                    // } --- scrapped idea
                }

                delete[] currRep;

                std::cout << std::endl;

                if (summandIndices.size() == 1) {
                    std::cout << "Set is Bracket-Dependent because this element is equal to another one in the set." << std::endl;
                    // bracketIndep = false;
                    continue;
                }

#if 0
                // DEAL WITH SUM HERE
                std::vector<int> workingSummands(summandIndices);

                for (auto summand : summandIndices) {

                    // std::cout << "Summand " << summand << std::endl;

                    bool keepGoing = true;
                    int sumSize = 0;

                    if (getRowSum(this->devGamma, this->orderDevGamma, this->orderDevGamma, summand, summandIndices.data(), summandIndices.size())) {
                        ++sumSize;
                        workingSummands.erase(
                            std::remove(workingSummands.begin(), workingSummands.end(), summand),
                            workingSummands.end()
                        );

                        // while(keepGoing) {
                        //     for(auto s : workingSummands) {

                        //         if(getRowSum(this->devGamma, this->orderDevGamma, this->orderDevGamma, s, workingSummands.data(), workingSummands.size())) {
                        //             ++sumSize;
                        //             workingSummands.erase(
                        //                 std::remove(workingSummands.begin(), workingSummands.end(), s),
                        //                 workingSummands.end()
                        //             );

                        //         }

                        //     }

                        // }
                        // std::cout << "TEST" << std::endl;

                    }


                    // std::cout << "Row sum " << getRowSum(this->devGamma, this->orderDevGamma, this->orderDevGamma, summand, summandIndices.data(), summandIndices.size()) << std::endl;

                }

                // bracketDep = false;

#endif
                // For now, just do all perms -- at least they have an easy exit cond.
                bool foundValidAdjointSequence = false;
                // std::sort(summandIndices.begin(), summandIndices.end()); -- already sorted -- not needed

                do {
                    std::vector<int> permutation(summandIndices); // A specific permutation
                    bool validSequence = true;

                    for (size_t step = 0; step < permutation.size() - 1; ++step) {

                        int currentSummand = permutation[step];

                        // Build the set of remaining summands (step+1 onward)
                        std::vector<int> remaining(permutation.begin() + step + 1, permutation.end());

                        if (remaining.empty()) {
                            validSequence = false;
                            break; // No one left to sum over
                        }

                        if (!getRowSum(this->devGamma, this->orderDevGamma, this->orderDevGamma,
                                        currentSummand, remaining.data(), static_cast<int>(remaining.size()))) {
                            validSequence = false;
                            break; // This permutation doesn't work
                        }
                    }

                    // Last summand (leaf node): nothing left to sum with, so it's invalid
                    // could allow it if there's no sum required anymore, tho
                    // final element can't sum over empty set, tho ??
                    // --> assume last element is a leaf and not allowed

                    if (validSequence) {
                        foundValidAdjointSequence = true;
                        std::cout << "Valid adjoint sequence found for alpha_" << hostNonPivotCols[i] << ": ";
                        for (auto idx : permutation) {
                            std::cout << "alpha_" << idx << " ";
                        }
                        std::cout << std::endl;
                        break;
                    }

                } while (std::next_permutation(summandIndices.begin(), summandIndices.end()));

                if (!foundValidAdjointSequence)
                    {   std::cout << "Not Valid adjoint sequence found for alpha_" << hostNonPivotCols[i] << " ";
                        int iter =hostNonPivotCols[i];
                        left_vec.push_back(iter);
                    }

            }





        }
        return left_vec;

    }



void MCP::generateProductGroup() {

            std::pair<uint32_t*, uint32_t*> temp = devGenerateProductGroup(this->devDecimalPx, this->devDecimalPz, this->rankF2Mat);
            this->devProductGroupPx = temp.first;
            this->devProductGroupPz = temp.second;

        }

void MCP::printProductGroup() {

            uint32_t cardProductGroup = 1 << this->rankF2Mat;

            uint32_t* hostPx = new uint32_t[cardProductGroup];
            uint32_t* hostPz = new uint32_t[cardProductGroup];

            CUDA_CHECK(cudaMemcpy(hostPx, this->devProductGroupPx, sizeof(uint32_t) * cardProductGroup, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hostPz, this->devProductGroupPz, sizeof(uint32_t) * cardProductGroup, cudaMemcpyDeviceToHost));

            // int count = 0;

            for(int i = 0; i < cardProductGroup; ++i) {
#if PRINT
                std::cout << decimalPairToPauliString(hostPx[i], hostPz[i], this->numberOfQubits) << " ";
#endif
                // ++count;
            }

            std::cout << std::endl;

            // std::cout << "COUNT" << count << std::endl;

            delete[] hostPx;
            delete[] hostPz;

        }


void MCP::printBracketClosure() {

            uint32_t cardProductGroup = 1 << this->rankF2Mat;

            uint32_t* hostPx = new uint32_t[cardProductGroup];
            uint32_t* hostPz = new uint32_t[cardProductGroup];

            CUDA_CHECK(cudaMemcpy(hostPx, this->devProductGroupPx, sizeof(uint32_t) * cardProductGroup, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hostPz, this->devProductGroupPz, sizeof(uint32_t) * cardProductGroup, cudaMemcpyDeviceToHost));


            uint32_t count = 0;

            for(int i = 0; i < cardProductGroup; ++i) {
                if (checkOdd(hostPx[i], hostPz[i], this->numberOfQubits)) {
#if PRINT
                    std::cout << decimalPairToPauliString(hostPx[i], hostPz[i], this->numberOfQubits) << " ";
#endif
                    ++count;
                }
            }

            std::cout << std::endl;

            std::cout << cardProductGroup << " " << count << std::endl;

            delete[] hostPx;
            delete[] hostPz;

        }

 void MCP::printPauliStrings() {
            for (int i = 0; i < this->cardinality; ++i) {
                std::cout << this->pauliStrings[i] << " ";
            }
            std::cout << " the cardinality is " << this-> cardinality << std::endl;

        };

std::vector<std::string> MCP::getPaulis(){
          return this->pauliStrings;
}

int MCP::getCardinality(){
    return this->cardinality;
}

int MCP::getBracketClosureSize() {
    uint32_t cardProductGroup = 1 << this->rankF2Mat;

    uint32_t* hostPx = new uint32_t[cardProductGroup];
    uint32_t* hostPz = new uint32_t[cardProductGroup];

    CUDA_CHECK(cudaMemcpy(hostPx, this->devProductGroupPx, sizeof(uint32_t) * cardProductGroup, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hostPz, this->devProductGroupPz, sizeof(uint32_t) * cardProductGroup, cudaMemcpyDeviceToHost));

    int count = 0;

    // Filter the set using your symmetry conditions
    for(int i = 0; i < cardProductGroup; ++i) {
        // Convert the binary pairs back to a std::string
        std::string pauli = decimalPairToPauliString(hostPx[i], hostPz[i], this->numberOfQubits);

        // Only count it if it satisfies BOTH the Odd and EvenFlips symmetries
        if (stringOdd(pauli, this->numberOfQubits) && stringEvenFlips(pauli, this->numberOfQubits)) {
            ++count;
        }
    }

    delete[] hostPx;
    delete[] hostPz;

    return count;
}