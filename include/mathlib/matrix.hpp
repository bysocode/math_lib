// mathlib/matrix.hpp
//
// A small, dependency-free dense matrix type.
//
// Storage is row-major and contiguous, so iteration over a Matrix<T> walks the
// elements in row-major order. Indexing with operator() is unchecked (it only
// asserts in debug builds); at() throws std::out_of_range.
//
// Errors:
//   std::invalid_argument  shape mismatches, ragged initializer lists, bad data size
//   std::out_of_range      at(), row(), col(), swapRows() with an invalid index
//   std::domain_error      an operation that needs a square matrix
//   std::runtime_error     inverse() of a singular matrix

#ifndef MATHLIB_MATRIX_HPP
#define MATHLIB_MATRIX_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace mathlib {

namespace detail {

template <typename T>
constexpr T absval(T value) noexcept {
    if constexpr (std::is_unsigned<T>::value) {
        return value;
    } else {
        return value < T{0} ? static_cast<T>(-value) : value;
    }
}

}  // namespace detail

template <typename T>
class Matrix {
public:
    using value_type = T;
    using size_type = std::size_t;
    using container_type = std::vector<T>;
    using iterator = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;
    using reference = T&;
    using const_reference = const T&;

    // ---- construction -------------------------------------------------

    Matrix() = default;

    Matrix(size_type rows, size_type cols)
        : rows_(rows), cols_(cols), data_(rows * cols) {}

    Matrix(size_type rows, size_type cols, const T& value)
        : rows_(rows), cols_(cols), data_(rows * cols, value) {}

    // Matrix<int> m{{1, 2}, {3, 4}};
    // Throws std::invalid_argument if the rows are not all the same length.
    Matrix(std::initializer_list<std::initializer_list<T>> rows) {
        rows_ = rows.size();
        cols_ = rows_ == 0 ? 0 : rows.begin()->size();
        data_.reserve(rows_ * cols_);
        for (const auto& row : rows) {
            if (row.size() != cols_) {
                throw std::invalid_argument(
                    "Matrix: initializer list rows must all have the same length");
            }
            data_.insert(data_.end(), row.begin(), row.end());
        }
    }

    // Build from an existing row-major buffer.
    static Matrix fromRowMajor(size_type rows, size_type cols, container_type values) {
        if (values.size() != rows * cols) {
            throw std::invalid_argument(
                "Matrix::fromRowMajor: data size does not match the given dimensions");
        }
        Matrix m;
        m.rows_ = rows;
        m.cols_ = cols;
        m.data_ = std::move(values);
        return m;
    }

    // ---- factories ----------------------------------------------------

    static Matrix zeros(size_type rows, size_type cols) { return Matrix(rows, cols); }

    static Matrix ones(size_type rows, size_type cols) { return Matrix(rows, cols, T{1}); }

    static Matrix identity(size_type n) {
        Matrix m(n, n);
        for (size_type i = 0; i < n; ++i) {
            m.data_[i * n + i] = T{1};
        }
        return m;
    }

    // ---- shape and access ---------------------------------------------

    size_type rows() const noexcept { return rows_; }
    size_type cols() const noexcept { return cols_; }
    size_type size() const noexcept { return data_.size(); }
    bool empty() const noexcept { return data_.empty(); }
    bool isSquare() const noexcept { return rows_ == cols_; }

    std::string dims() const {
        return std::to_string(rows_) + "x" + std::to_string(cols_);
    }

    // Unchecked in release builds.
    reference operator()(size_type row, size_type col) noexcept {
        assert(row < rows_ && col < cols_);
        return data_[row * cols_ + col];
    }

    const_reference operator()(size_type row, size_type col) const noexcept {
        assert(row < rows_ && col < cols_);
        return data_[row * cols_ + col];
    }

    reference at(size_type row, size_type col) {
        checkIndex(row, col, "Matrix::at");
        return data_[row * cols_ + col];
    }

    const_reference at(size_type row, size_type col) const {
        checkIndex(row, col, "Matrix::at");
        return data_[row * cols_ + col];
    }

    T* data() noexcept { return data_.data(); }
    const T* data() const noexcept { return data_.data(); }

    iterator begin() noexcept { return data_.begin(); }
    iterator end() noexcept { return data_.end(); }
    const_iterator begin() const noexcept { return data_.begin(); }
    const_iterator end() const noexcept { return data_.end(); }

    std::vector<T> row(size_type r) const {
        if (r >= rows_) {
            throw std::out_of_range("Matrix::row: row index out of range");
        }
        return std::vector<T>(data_.begin() + static_cast<std::ptrdiff_t>(r * cols_),
                              data_.begin() + static_cast<std::ptrdiff_t>((r + 1) * cols_));
    }

    std::vector<T> col(size_type c) const {
        if (c >= cols_) {
            throw std::out_of_range("Matrix::col: column index out of range");
        }
        std::vector<T> out(rows_);
        for (size_type r = 0; r < rows_; ++r) {
            out[r] = data_[r * cols_ + c];
        }
        return out;
    }

    // ---- mutators -----------------------------------------------------

    void fill(const T& value) { std::fill(data_.begin(), data_.end(), value); }

    void setZero() { fill(T{0}); }

    void swapRows(size_type a, size_type b) {
        if (a >= rows_ || b >= rows_) {
            throw std::out_of_range("Matrix::swapRows: row index out of range");
        }
        if (a == b) {
            return;
        }
        for (size_type c = 0; c < cols_; ++c) {
            std::swap(data_[a * cols_ + c], data_[b * cols_ + c]);
        }
    }

    // ---- element-wise arithmetic --------------------------------------

    Matrix& operator+=(const Matrix& other) {
        checkSameShape(other, "+=");
        for (size_type i = 0; i < data_.size(); ++i) {
            data_[i] += other.data_[i];
        }
        return *this;
    }

    Matrix& operator-=(const Matrix& other) {
        checkSameShape(other, "-=");
        for (size_type i = 0; i < data_.size(); ++i) {
            data_[i] -= other.data_[i];
        }
        return *this;
    }

    Matrix& operator*=(const T& scalar) {
        for (auto& value : data_) {
            value *= scalar;
        }
        return *this;
    }

    Matrix& operator/=(const T& scalar) {
        for (auto& value : data_) {
            value /= scalar;
        }
        return *this;
    }

    Matrix operator-() const {
        Matrix result = *this;
        for (auto& value : result.data_) {
            value = static_cast<T>(-value);
        }
        return result;
    }

    Matrix operator+(const Matrix& other) const {
        Matrix result = *this;
        result += other;
        return result;
    }

    Matrix operator-(const Matrix& other) const {
        Matrix result = *this;
        result -= other;
        return result;
    }

    Matrix operator*(const T& scalar) const {
        Matrix result = *this;
        result *= scalar;
        return result;
    }

    Matrix operator/(const T& scalar) const {
        Matrix result = *this;
        result /= scalar;
        return result;
    }

    // Element-wise (Hadamard) product.
    Matrix hadamard(const Matrix& other) const {
        checkSameShape(other, "hadamard");
        Matrix result = *this;
        for (size_type i = 0; i < result.data_.size(); ++i) {
            result.data_[i] *= other.data_[i];
        }
        return result;
    }

    // ---- linear algebra -----------------------------------------------

    // Matrix product; requires cols() == other.rows().
    Matrix operator*(const Matrix& other) const {
        if (cols_ != other.rows_) {
            throw std::invalid_argument(
                "Matrix: cannot multiply " + dims() + " by " + other.dims());
        }
        Matrix result(rows_, other.cols_);
        // i-k-j order keeps both the source row and the destination row hot.
        for (size_type i = 0; i < rows_; ++i) {
            for (size_type k = 0; k < cols_; ++k) {
                const T factor = data_[i * cols_ + k];
                if (factor == T{0}) {
                    continue;
                }
                for (size_type j = 0; j < other.cols_; ++j) {
                    result.data_[i * other.cols_ + j] += factor * other.data_[k * other.cols_ + j];
                }
            }
        }
        return result;
    }

    Matrix transpose() const {
        Matrix result(cols_, rows_);
        for (size_type i = 0; i < rows_; ++i) {
            for (size_type j = 0; j < cols_; ++j) {
                result.data_[j * rows_ + i] = data_[i * cols_ + j];
            }
        }
        return result;
    }

    T trace() const {
        if (!isSquare()) {
            throw std::domain_error("Matrix::trace: matrix must be square, got " + dims());
        }
        T sum = T{0};
        for (size_type i = 0; i < rows_; ++i) {
            sum += data_[i * cols_ + i];
        }
        return sum;
    }

    // LU decomposition with partial pivoting. Requires a floating point scalar.
    T determinant() const {
        static_assert(std::is_floating_point<T>::value,
                      "Matrix::determinant requires a floating point scalar type");
        if (!isSquare()) {
            throw std::domain_error("Matrix::determinant: matrix must be square, got " + dims());
        }
        const size_type n = rows_;
        if (n == 0) {
            return T{1};
        }

        container_type a = data_;
        const T tolerance = pivotTolerance();
        T det = T{1};

        for (size_type col = 0; col < n; ++col) {
            size_type pivot = col;
            T best = detail::absval(a[col * n + col]);
            for (size_type r = col + 1; r < n; ++r) {
                const T candidate = detail::absval(a[r * n + col]);
                if (candidate > best) {
                    best = candidate;
                    pivot = r;
                }
            }
            if (best <= tolerance) {
                return T{0};
            }
            if (pivot != col) {
                for (size_type c = 0; c < n; ++c) {
                    std::swap(a[col * n + c], a[pivot * n + c]);
                }
                det = static_cast<T>(-det);
            }
            const T diagonal = a[col * n + col];
            det *= diagonal;
            for (size_type r = col + 1; r < n; ++r) {
                const T factor = a[r * n + col] / diagonal;
                if (factor == T{0}) {
                    continue;
                }
                for (size_type c = col; c < n; ++c) {
                    a[r * n + c] -= factor * a[col * n + c];
                }
            }
        }
        return det;
    }

    // Gauss-Jordan elimination with partial pivoting.
    Matrix inverse() const {
        static_assert(std::is_floating_point<T>::value,
                      "Matrix::inverse requires a floating point scalar type");
        if (!isSquare()) {
            throw std::domain_error("Matrix::inverse: matrix must be square, got " + dims());
        }
        const size_type n = rows_;
        if (n == 0) {
            return Matrix();
        }

        container_type a = data_;
        container_type inv(n * n, T{0});
        for (size_type i = 0; i < n; ++i) {
            inv[i * n + i] = T{1};
        }
        const T tolerance = pivotTolerance();

        for (size_type col = 0; col < n; ++col) {
            size_type pivot = col;
            T best = detail::absval(a[col * n + col]);
            for (size_type r = col + 1; r < n; ++r) {
                const T candidate = detail::absval(a[r * n + col]);
                if (candidate > best) {
                    best = candidate;
                    pivot = r;
                }
            }
            if (best <= tolerance) {
                throw std::runtime_error("Matrix::inverse: matrix is singular");
            }
            if (pivot != col) {
                for (size_type c = 0; c < n; ++c) {
                    std::swap(a[col * n + c], a[pivot * n + c]);
                    std::swap(inv[col * n + c], inv[pivot * n + c]);
                }
            }
            const T diagonal = a[col * n + col];
            for (size_type c = 0; c < n; ++c) {
                a[col * n + c] /= diagonal;
                inv[col * n + c] /= diagonal;
            }
            for (size_type r = 0; r < n; ++r) {
                if (r == col) {
                    continue;
                }
                const T factor = a[r * n + col];
                if (factor == T{0}) {
                    continue;
                }
                for (size_type c = 0; c < n; ++c) {
                    a[r * n + c] -= factor * a[col * n + c];
                    inv[r * n + c] -= factor * inv[col * n + c];
                }
            }
        }
        return Matrix::fromRowMajor(n, n, std::move(inv));
    }

    // ---- norms and predicates -----------------------------------------

    double frobeniusNorm() const {
        double sum = 0.0;
        for (const auto& value : data_) {
            const double v = static_cast<double>(detail::absval(value));
            sum += v * v;
        }
        return std::sqrt(sum);
    }

    // Maximum absolute row sum (the matrix infinity norm).
    T normInf() const {
        T best = T{0};
        for (size_type r = 0; r < rows_; ++r) {
            T sum = T{0};
            for (size_type c = 0; c < cols_; ++c) {
                sum += detail::absval(data_[r * cols_ + c]);
            }
            if (sum > best) {
                best = sum;
            }
        }
        return best;
    }

    T maxAbs() const {
        T best = T{0};
        for (const auto& value : data_) {
            const T candidate = detail::absval(value);
            if (candidate > best) {
                best = candidate;
            }
        }
        return best;
    }

    bool isZero(T tolerance = defaultTolerance()) const {
        for (const auto& value : data_) {
            if (detail::absval(value) > tolerance) {
                return false;
            }
        }
        return true;
    }

    bool isApprox(const Matrix& other, T tolerance = defaultTolerance()) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            return false;
        }
        for (size_type i = 0; i < data_.size(); ++i) {
            if (detail::absval(data_[i] - other.data_[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }

    bool isSymmetric(T tolerance = defaultTolerance()) const {
        if (!isSquare()) {
            return false;
        }
        for (size_type r = 0; r < rows_; ++r) {
            for (size_type c = r + 1; c < cols_; ++c) {
                if (detail::absval(data_[r * cols_ + c] - data_[c * cols_ + r]) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    }

    bool isIdentity(T tolerance = defaultTolerance()) const {
        if (!isSquare()) {
            return false;
        }
        for (size_type r = 0; r < rows_; ++r) {
            for (size_type c = 0; c < cols_; ++c) {
                const T expected = (r == c) ? T{1} : T{0};
                if (detail::absval(data_[r * cols_ + c] - expected) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    }

    // Zero for integer scalars, since std::numeric_limits<int>::epsilon() == 0.
    static constexpr T defaultTolerance() noexcept {
        return std::numeric_limits<T>::epsilon() * static_cast<T>(100);
    }

    // ---- comparison ---------------------------------------------------

    friend bool operator==(const Matrix& lhs, const Matrix& rhs) {
        return lhs.rows_ == rhs.rows_ && lhs.cols_ == rhs.cols_ && lhs.data_ == rhs.data_;
    }

    friend bool operator!=(const Matrix& lhs, const Matrix& rhs) { return !(lhs == rhs); }

private:
    void checkIndex(size_type row, size_type col, const char* what) const {
        if (row >= rows_ || col >= cols_) {
            throw std::out_of_range(std::string(what) + ": index (" + std::to_string(row) + ", " +
                                    std::to_string(col) + ") out of range for " + dims());
        }
    }

    void checkSameShape(const Matrix& other, const char* op) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument(std::string("Matrix: shape mismatch in ") + op + " (" +
                                        dims() + " vs " + other.dims() + ")");
        }
    }

    T pivotTolerance() const {
        const T scale = std::max<T>(maxAbs(), T{1});
        const T dimension = static_cast<T>(std::max<size_type>(rows_, 1));
        return std::numeric_limits<T>::epsilon() * dimension * scale * static_cast<T>(8);
    }

    size_type rows_ = 0;
    size_type cols_ = 0;
    container_type data_;
};

// ---- free functions ---------------------------------------------------

template <typename T>
Matrix<T> operator*(const T& scalar, const Matrix<T>& matrix) {
    return matrix * scalar;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix) {
    if (matrix.rows() == 0) {
        os << "[]";
        return os;
    }

    std::vector<std::vector<std::string>> cells(
        matrix.rows(), std::vector<std::string>(matrix.cols()));
    std::vector<std::size_t> widths(matrix.cols(), 0);

    for (std::size_t r = 0; r < matrix.rows(); ++r) {
        for (std::size_t c = 0; c < matrix.cols(); ++c) {
            std::ostringstream cell;
            cell << matrix(r, c);
            cells[r][c] = cell.str();
            widths[c] = std::max(widths[c], cells[r][c].size());
        }
    }

    for (std::size_t r = 0; r < matrix.rows(); ++r) {
        os << '[';
        for (std::size_t c = 0; c < matrix.cols(); ++c) {
            if (c != 0) {
                os << "  ";
            }
            os << std::string(widths[c] - cells[r][c].size(), ' ') << cells[r][c];
        }
        os << ']';
        if (r + 1 < matrix.rows()) {
            os << '\n';
        }
    }
    return os;
}

template <typename T>
std::string to_string(const Matrix<T>& matrix) {
    std::ostringstream os;
    os << matrix;
    return os.str();
}

}  // namespace mathlib

#endif  // MATHLIB_MATRIX_HPP
