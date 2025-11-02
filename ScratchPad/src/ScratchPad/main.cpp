#include <Argon.h>
#include <iostream>

int main()
{
    
    std::cout << "SIMD Size: " << ARGON_SIMD_SIZE << "\n";

    // floats:
    argon::vec2f vec2fA{1, 2};
    argon::vec2f vec2fB{3, 4};

    std::cout << "Floats:\n";
    std::cout << "add: \t" << vec2fA + vec2fB << "\n";
    std::cout << "sub: \t" << vec2fA - vec2fB << "\n";
    std::cout << "mul: \t" << vec2fA * vec2fB << "\n";
    std::cout << "div: \t" << vec2fA / vec2fB << "\n";
    std::cout << "dot: \t" << argon::Dot(vec2fA, vec2fB) << "\n";
    std::cout << "A length: " << vec2fA.length() << " B length: " << vec2fB.length() << "\n";

    std::cout << std::endl;

    // doubles
    argon::vec4d vec4dA{3, 6, 2, 9};
    argon::vec4d vec4dB{0.56, 9.2, 3.14159, 12};

    std::cout << "Doubles:\n";
    std::cout << "add: \t" << vec4dA + vec4dB << "\n";
    std::cout << "sub: \t" << vec4dA - vec4dB << "\n";
    std::cout << "mul: \t" << vec4dA * vec4dB << "\n";
    std::cout << "div: \t" << vec4dA / vec4dB << "\n";
    std::cout << "dot: \t" << argon::Dot(vec4dA, vec4dB) << "\n";
    std::cout << "A length: " << vec4dA.length() << " B length: " << vec4dB.length() << "\n";

    std::cout << std::endl;

    return 0;
}