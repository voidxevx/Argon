#ifndef ARGON_MATRIX
#define ARGON_MATRIX

#include "ArgonSIMD.h"
#include "Vector.h"

namespace argon
{

    template<typename _T, size_t _rows, size_t _cols>
    struct 
    Matrix
    {
        _T items[_rows * _cols];

        constexpr 
        Matrix()
        {
            for (size_t i{}; i < _rows * _cols; ++i)
                items[i] = _T{0};
        }
        constexpr
        Matrix(std::initializer_list<_T> values)
        {
            size_t i{};
            for (auto val : values)
                items[i++] = val;
            for (; i < _rows * _cols; ++i)
                items[i] = _T{0};
        }

        constexpr const _T &operator()(size_t row, size_t col) const { return items[row + col * _cols]; }
        constexpr _T &operator()(size_t row, size_t col) { return items[row + col * _cols]; }


        constexpr Vector<_T, _cols>
        GetRow(size_t col)
        const
        {
            Vector<_T, _cols> ret;
            for (size_t i{}; i < _cols; ++i)
                ret[i] = items[i + col * _cols];
            return ret;
        }

        constexpr Vector<_T, _rows>
        GetCol(size_t row)
        const
        {
            Vector<_T, _rows> ret;
            for (size_t i{}; i < _rows; ++i)
                ret[i] = items[row + i * _cols];
            return ret;
        }

        constexpr Matrix<_T, _rows, _cols>
        operator*(const Matrix<_T, _cols, _rows> &other)
        const
        {
            Matrix<_T, _rows, _cols> ret;
            matrix_mul(*this, other, ret);
            return ret;
        }

        constexpr Vector<_T, _rows>
        operator*(const Vector<_T, _cols> &other)
        const
        {
            Vector<_T, _rows> ret;
            vec_transform(*this, other, ret);
            return ret;
        }

        static constexpr Matrix<_T, _rows, _cols>
        Identity()
        {
            Matrix<_T, _rows, _cols> ret;
            for (size_t i{}; i < _rows; ++i)
                if (i < _cols)
                    ret(i, i) = _T{1};
            return ret;
        }

        static constexpr Matrix<_T, _rows, _cols>
        Scalar(const Vector<_T, std::min(_rows, _cols)> &base)
        {
            Matrix<_T, _rows, _cols> ret;
            for (size_t i{}; i < _rows; ++i)
                if (i < _cols)
                    ret(i, i) = base.items[i];
            return ret;
        }

        static constexpr Matrix<_T, _rows, _cols>
        Translation(const Vector<_T, _rows - 1> &base)
        {
            Matrix<_T, _rows, _cols> ret = Identity();
            for (size_t i{}; i < _rows - 1; ++i)
                ret(_rows - 1, i) = base.items[i];
            return ret;
        }

    };

    template<typename _T, size_t _rows, size_t _cols>
    inline std::ostream 
    &operator<<(std::ostream &stream, const Matrix<_T, _rows, _cols> &mat)
    {
        for (size_t col{}; col < _cols; ++col)
        {
            stream << col << ": | ";
            for (size_t row{}; row < _rows; ++row)
            {
                stream << mat(row, col) << " ";
            }
            stream << "| ";
        }
    }

    template<typename _T, size_t _rows, size_t _cols>
    inline void
    matrix_mul(const Matrix<_T, _rows, _cols> &A, const Matrix<_T, _rows, _cols> &B, Matrix<_T, _cols, _rows> &ret)
    {
        for (size_t row{}; row < _rows; ++row)
        {
            Vector<_T, _cols> c_row = A.GetRow(row);
            for (size_t col{}; col < _cols; ++col)
                ret(row, col) = Dot(c_row, B.GetCol(col));
        }
    }

    template<typename _T, size_t _rows, size_t _cols>
    inline void
    vec_transform(const Matrix<_T, _rows, _cols> &mat, const Vector<_T, _cols> &vec, Vector<_T, _rows> &ret)
    {
        for (size_t i{}; i < _rows; ++i)
            ret[i] = Dot(mat.GetRow(i), vec);
    }
    

}

#endif