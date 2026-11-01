// Tests for mathlib::Matrix. No external dependencies: a tiny check harness is
// defined below so the suite can run anywhere a C++17 compiler exists.

#include <mathlib/matrix.hpp>

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int g_checks = 0;
int g_failures = 0;

void reportFailure(const std::string& what, const char* file, int line) {
    ++g_failures;
    std::cerr << "    FAIL " << file << ':' << line << "  " << what << '\n';
}

#define CHECK(expr)                                                        \
    do {                                                                   \
        ++g_checks;                                                        \
        if (!(expr)) reportFailure(#expr, __FILE__, __LINE__);              \
    } while (false)

template <typename Exception, typename Fn>
bool throwsAs(Fn&& fn) {
    try {
        fn();
    } catch (const Exception&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

#define CHECK_THROWS_AS(expr, exception_type)                                     \
    do {                                                                          \
        ++g_checks;                                                               \
        if (!throwsAs<exception_type>([&] { (void)(expr); })) {                    \
            reportFailure("expected " #exception_type " from: " #expr, __FILE__,   \
                          __LINE__);                                              \
        }                                                                         \
    } while (false)

using mathlib::Matrix;
using M = Matrix<double>;

// ---- creation ----------------------------------------------------------

void testCreation() {
    const Matrix<double> empty;
    CHECK(empty.rows() == 0);
    CHECK(empty.cols() == 0);
    CHECK(empty.size() == 0);
    CHECK(empty.empty());
    CHECK(empty.isSquare());

    const M zero(2, 3);
    CHECK(zero.rows() == 2);
    CHECK(zero.cols() == 3);
    CHECK(zero.size() == 6);
    CHECK(!zero.empty());
    CHECK(zero.dims() == "2x3");
    CHECK(zero.isZero());

    const M filled(2, 2, 7.5);
    CHECK(filled(0, 0) == 7.5);
    CHECK(filled(1, 1) == 7.5);
    CHECK(filled.size() == 4);

    const M list{{1.0, 2.0}, {3.0, 4.0}};
    CHECK(list.rows() == 2);
    CHECK(list.cols() == 2);
    CHECK(list(0, 0) == 1.0);
    CHECK(list(0, 1) == 2.0);
    CHECK(list(1, 0) == 3.0);
    CHECK(list(1, 1) == 4.0);

    const Matrix<int> singleRow{{1, 2, 3}};
    CHECK(singleRow.rows() == 1);
    CHECK(singleRow.cols() == 3);

    CHECK_THROWS_AS(Matrix<int>({{1, 2}, {3}}), std::invalid_argument);

    CHECK(M::zeros(2, 2).isZero());
    CHECK(M::ones(2, 3)(1, 2) == 1.0);
    CHECK(!M::ones(2, 2).isZero());

    const M id = M::identity(3);
    CHECK(id(0, 0) == 1.0);
    CHECK(id(1, 1) == 1.0);
    CHECK(id(2, 2) == 1.0);
    CHECK(id(0, 1) == 0.0);
    CHECK(id.isIdentity());
    CHECK(M::identity(0).empty());

    const M copy = list;
    CHECK(copy == list);
    M moved = copy;
    const M movedInto = std::move(moved);
    CHECK(movedInto == list);

    const M built = M::fromRowMajor(2, 2, {1.0, 2.0, 3.0, 4.0});
    CHECK(built == list);
    CHECK_THROWS_AS(M::fromRowMajor(2, 2, {1.0, 2.0, 3.0}), std::invalid_argument);
}

// ---- element access ----------------------------------------------------

void testAccess() {
    M m{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

    CHECK(m(1, 2) == 6.0);
    m(1, 2) = 60.0;
    CHECK(m(1, 2) == 60.0);

    CHECK(m.at(0, 0) == 1.0);
    CHECK_THROWS_AS(m.at(2, 0), std::out_of_range);
    CHECK_THROWS_AS(m.at(0, 3), std::out_of_range);

    const std::vector<double> firstRow = m.row(1);
    CHECK(firstRow.size() == 3);
    CHECK(firstRow[2] == 60.0);

    const std::vector<double> secondCol = m.col(1);
    CHECK(secondCol.size() == 2);
    CHECK(secondCol[0] == 2.0);
    CHECK(secondCol[1] == 5.0);

    CHECK_THROWS_AS(m.row(5), std::out_of_range);
    CHECK_THROWS_AS(m.col(9), std::out_of_range);

    // Row-major iteration.
    double sum = 0.0;
    for (double value : m) {
        sum += value;
    }
    CHECK(sum == 1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 60.0);

    CHECK(m.data() != nullptr);
    CHECK(m.data()[0] == 1.0);
    CHECK(m.data()[5] == 60.0);

    m.swapRows(0, 1);
    CHECK(m(0, 0) == 4.0);
    CHECK(m(1, 0) == 1.0);
    CHECK_THROWS_AS(m.swapRows(0, 7), std::out_of_range);

    m.fill(2.0);
    CHECK(m.isZero() == false);
    CHECK(m(0, 0) == 2.0);
    m.setZero();
    CHECK(m.isZero());
}

// ---- element-wise arithmetic -------------------------------------------

void testArithmetic() {
    const M a{{1.0, 2.0}, {3.0, 4.0}};
    const M b{{10.0, 20.0}, {30.0, 40.0}};

    CHECK((a + b) == M({{11.0, 22.0}, {33.0, 44.0}}));
    CHECK((a - b) == M({{-9.0, -18.0}, {-27.0, -36.0}}));
    CHECK((a * 2.0) == M({{2.0, 4.0}, {6.0, 8.0}}));
    CHECK((2.0 * a) == M({{2.0, 4.0}, {6.0, 8.0}}));
    CHECK((a / 2.0) == M({{0.5, 1.0}, {1.5, 2.0}}));
    CHECK(-a == M({{-1.0, -2.0}, {-3.0, -4.0}}));

    M c = a;
    c += b;
    CHECK(c == a + b);
    c -= b;
    CHECK(c == a);
    c *= 3.0;
    CHECK(c == M({{3.0, 6.0}, {9.0, 12.0}}));
    c /= 3.0;
    CHECK(c == a);

    CHECK(a.hadamard(b) == M({{10.0, 40.0}, {90.0, 160.0}}));

    CHECK_THROWS_AS(a + M({{1.0, 2.0, 3.0}}), std::invalid_argument);
    CHECK_THROWS_AS(a - M({{1.0}}), std::invalid_argument);
    CHECK_THROWS_AS(a.hadamard(M({{1.0}})), std::invalid_argument);
    CHECK_THROWS_AS(c += M({{1.0, 2.0, 3.0}}), std::invalid_argument);
}

// ---- matrix multiplication ---------------------------------------------

void testMultiply() {
    const M a{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const M b{{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};

    const M product = a * b;
    CHECK(product.rows() == 2);
    CHECK(product.cols() == 2);
    CHECK(product == M({{58.0, 64.0}, {139.0, 154.0}}));

    CHECK_THROWS_AS(a * a, std::invalid_argument);

    CHECK(M({{1.0, 2.0, 3.0}}) * M::identity(3) == M({{1.0, 2.0, 3.0}}));
    CHECK(M::identity(2) * a == a);

    // Matrix multiplication does not commute.
    const M x{{1.0, 1.0}, {0.0, 1.0}};
    const M y{{1.0, 0.0}, {1.0, 1.0}};
    CHECK(x * y == M({{2.0, 1.0}, {1.0, 1.0}}));
    CHECK(y * x == M({{1.0, 1.0}, {1.0, 2.0}}));
    CHECK((x * y) != (y * x));

    CHECK((M() * M()).empty());
    CHECK((a * M::zeros(3, 2)).isZero());
}

// ---- transpose and trace ----------------------------------------------

void testTransposeTrace() {
    const M a{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

    const M t = a.transpose();
    CHECK(t.rows() == 3);
    CHECK(t.cols() == 2);
    CHECK(t == M({{1.0, 4.0}, {2.0, 5.0}, {3.0, 6.0}}));
    CHECK(a.transpose().transpose() == a);

    CHECK(M({{1.0, 2.0}, {3.0, 4.0}}).trace() == 5.0);
    CHECK(M::identity(4).trace() == 4.0);
    CHECK_THROWS_AS(a.trace(), std::domain_error);
}

// ---- determinant -------------------------------------------------------

void testDeterminant() {
    CHECK(std::abs(M({{1.0, 2.0}, {3.0, 4.0}}).determinant() - (-2.0)) < 1e-12);
    CHECK(std::abs(M({{5.0}}).determinant() - 5.0) < 1e-12);
    CHECK(M().determinant() == 1.0);  // empty product
    CHECK(std::abs(M::identity(4).determinant() - 1.0) < 1e-12);

    const M a3{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 10.0}};
    CHECK(std::abs(a3.determinant() - (-3.0)) < 1e-12);
    CHECK(std::abs(a3.transpose().determinant() - a3.determinant()) < 1e-12);

    const M b{{1.0, 2.0, 3.0}, {0.0, 1.0, 4.0}, {5.0, 6.0, 0.0}};
    CHECK(std::abs(b.determinant() - 1.0) < 1e-12);

    // Singular: second row is a multiple of the first.
    CHECK(M({{1.0, 2.0}, {2.0, 4.0}}).determinant() == 0.0);
    CHECK(M({{1.0, 2.0, 3.0}, {2.0, 4.0, 6.0}, {1.0, 1.0, 1.0}}).determinant() == 0.0);

    CHECK_THROWS_AS(M({{1.0, 2.0, 3.0}}).determinant(), std::domain_error);

    // Scaling a row scales the determinant.
    const M scaled = a3;
    CHECK(std::abs((scaled * 2.0).determinant() - 8.0 * a3.determinant()) < 1e-9);
}

// ---- inverse -----------------------------------------------------------

void testInverse() {
    const M a{{1.0, 2.0}, {3.0, 4.0}};
    const M ai = a.inverse();
    CHECK(ai.isApprox(M({{-2.0, 1.0}, {1.5, -0.5}})));
    CHECK((a * ai).isApprox(M::identity(2)));
    CHECK((ai * a).isApprox(M::identity(2)));

    const M b{{1.0, 2.0, 3.0}, {0.0, 1.0, 4.0}, {5.0, 6.0, 0.0}};
    const M bi = b.inverse();
    CHECK((b * bi).isApprox(M::identity(3)));
    CHECK((bi * b).isApprox(M::identity(3)));
    CHECK(bi.isApprox(b.inverse()));
    CHECK(std::abs(bi.determinant() - 1.0 / b.determinant()) < 1e-12);

    CHECK(M::identity(5).inverse().isApprox(M::identity(5)));
    CHECK(M().inverse().empty());

    // A diagonal matrix inverts element-wise.
    const M diagonal{{2.0, 0.0}, {0.0, 4.0}};
    CHECK(diagonal.inverse().isApprox(M({{0.5, 0.0}, {0.0, 0.25}})));

    CHECK_THROWS_AS(M({{1.0, 2.0}, {2.0, 4.0}}).inverse(), std::runtime_error);
    CHECK_THROWS_AS(M({{1.0, 2.0, 3.0}}).inverse(), std::domain_error);
}

// ---- norms and predicates ---------------------------------------------

void testNormsAndPredicates() {
    const M a{{3.0, -4.0}, {0.0, 5.0}};
    CHECK(std::abs(a.frobeniusNorm() - std::sqrt(50.0)) < 1e-12);
    CHECK(a.normInf() == 7.0);  // max(|3| + |-4|, |0| + |5|)
    CHECK(a.maxAbs() == 5.0);

    CHECK(M().frobeniusNorm() == 0.0);
    CHECK(M().normInf() == 0.0);
    CHECK(M().maxAbs() == 0.0);

    CHECK(M({{1.0, 2.0}, {2.0, 3.0}}).isSymmetric());
    CHECK(!M({{1.0, 2.0}, {3.0, 4.0}}).isSymmetric());
    CHECK(!M({{1.0, 2.0, 3.0}}).isSymmetric());
    CHECK(M().isSymmetric());

    CHECK(M({{1e-14}}).isZero());
    CHECK(!M({{1.0}}).isZero());
    CHECK(!M({{1.0, 0.0}}).isZero());

    CHECK(M::identity(3).isIdentity());
    CHECK(!M({{1.0, 0.0}, {0.0, 2.0}}).isIdentity());
    CHECK(!M({{1.0, 2.0, 3.0}}).isIdentity());

    CHECK(M({{1.0, 2.0}, {3.0, 4.0}}).isApprox(M({{1.0, 2.0}, {3.0, 4.0}})));
    CHECK(!M({{1.0, 2.0}, {3.0, 4.0}}).isApprox(M({{1.0, 2.0}, {3.0, 5.0}})));
    CHECK(!M({{1.0, 2.0}}).isApprox(M({{1.0, 2.0}, {3.0, 4.0}})));

    const M one{{1.0}};
    CHECK(one.isApprox(M({{1.0 + 1e-15}})));
    CHECK(!one.isApprox(M({{1.1}})));
    CHECK(one.isApprox(M({{1.1}}), 0.2));  // explicit tolerance

    CHECK(M({{1.0, 2.0}, {3.0, 4.0}}) == M({{1.0, 2.0}, {3.0, 4.0}}));
    CHECK(M({{1.0, 2.0}, {3.0, 4.0}}) != M({{1.0, 2.0}, {3.0, 5.0}}));
}

// ---- integer scalars ---------------------------------------------------

void testIntegerMatrix() {
    const Matrix<int> a{{1, 2}, {3, 4}};
    const Matrix<int> b{{5, 6}, {7, 8}};

    CHECK(a * b == Matrix<int>({{19, 22}, {43, 50}}));
    CHECK(a.trace() == 5);
    CHECK(a.maxAbs() == 4);
    CHECK(a.normInf() == 7);  // max(|1| + |2|, |3| + |4|)
    CHECK((a + b) == Matrix<int>({{6, 8}, {10, 12}}));
    CHECK((a + b).isApprox(Matrix<int>({{6, 8}, {10, 12}})));

    Matrix<int> negative{{-3, 2}};
    CHECK(negative.maxAbs() == 3);
    CHECK(-negative == Matrix<int>({{3, -2}}));
}

// ---- printing ----------------------------------------------------------

void testPrinting() {
    std::ostringstream plain;
    plain << M({{1.0, 2.0}, {3.0, 4.0}});
    CHECK(plain.str() == "[1  2]\n[3  4]");

    std::ostringstream aligned;
    aligned << M({{1.0, 10.0}, {100.0, 2.0}});
    CHECK(aligned.str() == "[  1  10]\n[100   2]");

    std::ostringstream empty;
    empty << M();
    CHECK(empty.str() == "[]");

    std::ostringstream wide;
    wide << M({{1.5}});
    CHECK(wide.str() == "[1.5]");

    CHECK(mathlib::to_string(M({{1.0, -2.0}})) == "[1  -2]");
}

struct TestCase {
    const char* name;
    void (*fn)();
};

const TestCase kTests[] = {
    {"creation", testCreation},
    {"element access", testAccess},
    {"element-wise arithmetic", testArithmetic},
    {"matrix multiplication", testMultiply},
    {"transpose and trace", testTransposeTrace},
    {"determinant", testDeterminant},
    {"inverse", testInverse},
    {"norms and predicates", testNormsAndPredicates},
    {"integer scalars", testIntegerMatrix},
    {"printing", testPrinting},
};

}  // namespace

int main() {
    for (const auto& test : kTests) {
        const int failuresBefore = g_failures;
        std::cout << "[ RUN  ] " << test.name << '\n';
        try {
            test.fn();
        } catch (const std::exception& e) {
            reportFailure(std::string("unexpected exception: ") + e.what(), __FILE__, __LINE__);
        } catch (...) {
            reportFailure("unexpected non-standard exception", __FILE__, __LINE__);
        }
        const bool ok = g_failures == failuresBefore;
        std::cout << (ok ? "[  OK  ] " : "[ FAIL ] ") << test.name << '\n';
    }

    std::cout << '\n'
              << g_checks << " checks, " << g_failures << " failure(s)" << '\n';
    return g_failures == 0 ? 0 : 1;
}
