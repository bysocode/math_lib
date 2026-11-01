# mathlib

A small, dependency-free dense matrix library for C++17.

- Header-only core: `include/mathlib/matrix.hpp`
- Row-major, contiguous storage in a `std::vector`
- Templated on the scalar type (`double`, `float`, `int`, ...)
- CMake build with a test suite and a demo

## Layout

```
include/mathlib/matrix.hpp   Matrix<T> implementation (the whole library)
src/matrix.cpp               explicit instantiations for float/double
tests/test_matrix.cpp        self-contained test suite
examples/demo.cpp            walkthrough of the API
CMakeLists.txt               build, tests, examples
```

## Build and test

```sh
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/mathlib_demo
```

To use the header without CMake, just add `-Iinclude` (no linking needed):

```sh
g++ -std=c++17 -Iinclude examples/demo.cpp -o demo
```

CMake options: `-DMATHLIB_BUILD_TESTS=OFF`, `-DMATHLIB_BUILD_EXAMPLES=OFF`.

## Quick example

```cpp
#include <mathlib/matrix.hpp>
#include <iostream>

int main() {
    using mathlib::Matrix;

    Matrix<double> a{{4.0, 7.0}, {2.0, 6.0}};
    Matrix<double> b{{1.0, 2.0}, {3.0, 4.0}};

    std::cout << a * b << '\n';              // matrix product
    std::cout << a.hadamard(b) << '\n';      // element-wise product
    std::cout << a.transpose() << '\n';
    std::cout << a.determinant() << '\n';    // 10
    std::cout << a.inverse() << '\n';
    std::cout << a.frobeniusNorm() << '\n';
}
```

Output of `operator<<` is aligned per column:

```
[25  36]
[20  28]
```

## API overview

| Group | Members |
| --- | --- |
| Creation | `Matrix()`, `Matrix(rows, cols)`, `Matrix(rows, cols, value)`, `Matrix{{1, 2}, {3, 4}}`, `zeros`, `ones`, `identity`, `fromRowMajor` |
| Shape | `rows()`, `cols()`, `size()`, `empty()`, `isSquare()`, `dims()` |
| Access | `operator()(r, c)`, `at(r, c)`, `data()`, `begin()`/`end()`, `row(r)`, `col(c)` |
| Mutate | `fill`, `setZero`, `swapRows` |
| Arithmetic | `+`, `-`, unary `-`, `*`/`/` by scalar, `+=`, `-=`, `*=`, `/=`, `hadamard` |
| Products | `operator*` (matrix), `transpose`, `trace` |
| Linear algebra | `determinant` (LU, partial pivoting), `inverse` (Gauss-Jordan) |
| Norms | `frobeniusNorm()`, `normInf()`, `maxAbs()` |
| Predicates | `==`, `!=`, `isApprox`, `isZero`, `isSymmetric`, `isIdentity` |
| I/O | `operator<<`, `mathlib::to_string` |

## Behavior notes

- `operator()(r, c)` is unchecked and only `assert`s in debug builds; `at()`
  throws `std::out_of_range`.
- Shape mismatches throw `std::invalid_argument`, including ragged
  initializer lists and a wrong-size `fromRowMajor` buffer.
- `trace`, `determinant`, and `inverse` require a square matrix and throw
  `std::domain_error` otherwise. `inverse` throws `std::runtime_error` for a
  singular matrix.
- `determinant` and `inverse` are `static_assert`-restricted to floating point
  scalars; everything else works for any arithmetic type.
- `==` compares exactly. For floating point results use `isApprox`, whose
  default tolerance is `100 * epsilon` of the scalar type (zero for integers,
  i.e. exact comparison).
- Matrix multiplication uses an `i-k-j` loop order and skips zero factors.
- Pivoting uses a scale-aware tolerance so near-singular matrices are detected
  rather than producing garbage.

## Limitations

- Dense only, no sparse support or views/expressions (operations allocate).
- No decompositions beyond LU/Gauss-Jordan (no QR, SVD, Cholesky).
- No parallelization or SIMD; intended for small matrices and teaching.
