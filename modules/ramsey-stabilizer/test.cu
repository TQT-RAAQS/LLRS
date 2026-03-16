#include <iostream>
#include "fourier-analyzer.h"
#include <fstream>
#include <vector>
#include <cstdint>

int main() {
    auto fa = FourierAnalyzer(128, 128);

    size_t array_size = 29 * 23;
    std::vector<uint8_t> a0(array_size), a1(array_size);

    std::ifstream f0("/home/tqtraaqs/Desktop/tqtraaqs_git/LLRS/modules/ramsey-stabilizer/a0.bin", std::ios::binary);
    std::ifstream f1("/home/tqtraaqs/Desktop/tqtraaqs_git/LLRS/modules/ramsey-stabilizer/a1.bin", std::ios::binary);

    f0.read(reinterpret_cast<char*>(a0.data()), array_size);
    f1.read(reinterpret_cast<char*>(a1.data()), array_size);

    auto phi = fa.extract_phase(a0, a1);

    std::cout << phi << std::endl;
    
    return 0;
}