// The library is header-only by design: everything lives in
// <mathlib/matrix.hpp> so it can be dropped into another project directly.
//
// This translation unit exists so the build produces a real library target and
// so the templates are instantiated (and therefore fully type-checked) for the
// scalar types we care about.

#include <mathlib/matrix.hpp>

namespace mathlib {

template class Matrix<float>;
template class Matrix<double>;

}  // namespace mathlib
