#include <Argon.h>
#include <iostream>

int main()
{
    
    std::cout << "SIMD Size: " << ARGON_SIMD_SIZE << "\n";

    // floats:
    argon::vec3 vecA{1, 2, 5};
    argon::vec3 vecB{3, 4, 6};

    std::cout << "Floats:\n";
    std::cout << "add: \t" << vecA + vecB << "\n";
    std::cout << "sub: \t" << vecA - vecB << "\n";
    std::cout << "mul: \t" << vecA * vecB << "\n";
    std::cout << "div: \t" << vecA / vecB << "\n";
    std::cout << "dot: \t" << argon::Dot(vecA, vecB) << "\n";
    std::cout << "A length: " << vecA.length() << " B length: " << vecB.length() << "\n";

    std::cout << std::endl;

    argon::Matrix<float, 4, 4> matA = argon::Matrix<float, 4, 4>::Translation(vecA);
    std::cout << matA << std::endl;

    return 0;
}