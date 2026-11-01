// Small tour of mathlib::Matrix: creation, arithmetic, multiplication,
// transpose, determinant, inverse, norms, and solving a linear system.

#include <mathlib/matrix.hpp>

#include <iostream>

int main() {
    using mathlib::Matrix;

    const Matrix<double> a{{4.0, 7.0}, {2.0, 6.0}};
    const Matrix<double> b{{1.0, 2.0}, {3.0, 4.0}};

    std::cout << "A =\n" << a << "\n\n";
    std::cout << "B =\n" << b << "\n\n";

    std::cout << "A + B =\n" << a + b << "\n\n";
    std::cout << "A - B =\n" << a - b << "\n\n";
    std::cout << "2 * A =\n" << a * 2.0 << "\n\n";
    std::cout << "A * B (matrix product) =\n" << a * b << "\n\n";
    std::cout << "A .* B (Hadamard) =\n" << a.hadamard(b) << "\n\n";

    std::cout << "A^T =\n" << a.transpose() << "\n\n";

    std::cout << "det(A) = " << a.determinant() << '\n';
    std::cout << "trace(A) = " << a.trace() << '\n';
    std::cout << "||A||_F = " << a.frobeniusNorm() << '\n';
    std::cout << "||A||_inf = " << a.normInf() << "\n\n";

    const Matrix<double> aInv = a.inverse();
    std::cout << "A^-1 =\n" << aInv << "\n\n";
    std::cout << "A * A^-1 =\n" << a * aInv << '\n';
    std::cout << "A * A^-1 is identity: " << (a * aInv).isIdentity() << "\n\n";

    // Solve A x = rhs by x = A^-1 rhs.
    const Matrix<double> rhs{{1.0}, {0.0}};
    const Matrix<double> x = a.inverse() * rhs;
    std::cout << "solve A x = [1, 0]^T -> x =\n" << x << '\n';
    std::cout << "residual A x - rhs =\n" << a * x - rhs << '\n';

    // Factories and a non-square example.
    std::cout << "\nidentity(3) =\n" << Matrix<double>::identity(3) << "\n\n";

    const Matrix<double> tall = Matrix<double>::ones(3, 2) * 5.0;
    std::cout << "ones(3, 2) * 5 =\n" << tall << '\n';
    std::cout << "shape: " << tall.dims() << ", square: " << tall.isSquare() << '\n';

    return 0;
}
