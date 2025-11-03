#ifndef ARGON_VECTOR
#define ARGON_VECTOR

#include "ArgonSIMD.h"

#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <cmath>

namespace argon
{

    template <typename _T, size_t _l>
    struct
    #ifdef ARGON_SIMD_ENABLED 
    alignas(simd::SimdAlignment<_T>::value)
    #endif
    Vector
    {
        #ifdef ARGON_SIMD_ENABLED
        static constexpr size_t SIMDPadding = (_l < 4) ? 4 : _l;
        _T items[SIMDPadding];
        #else
        _T items[_l];
        #endif

        #ifdef ARGON_SIMD_ENABLED
        constexpr
        Vector() 
        {
            for (size_t i{}; i < SIMDPadding; ++i)
                items[i] = _T{0};
        }
        constexpr
        Vector(std::initializer_list<_T> values)
        {
            size_t i{};
            for (auto val : values)
                if (i < _l) items[i++] = val;
            for(; i < SIMDPadding; ++i)
                items[i] = _T{0};
        }
        #else
        constexpr
        Vector()
        {
            for (size_t i{}; i < _l; ++i)
                items[i] = _T{0};
        }
        constexpr
        Vector(std::initializer_list<_T> values)
        {
            size_t i{};
            for (auto val : values)
                if (i < _l) items[i++] = val;
        }
        #endif

        constexpr _T& operator[](size_t i) { return this->items[i]; }
        constexpr const _T& operator[](size_t i) const { return this->items[i]; }


        inline Vector<_T, _l> 
        operator+(const Vector<_T, _l>& other) 
        const
        {
            Vector<_T, _l> ret;
            #ifdef ARGON_SIMD_ENABLED
            vec_add_simd(*this, other, ret);
            #else
            for (size_t i{}; i < _l; ++i)
                ret.items[i] = items[i] + other.items[i];
            #endif
            return ret;
        }

        inline Vector<_T, _l>
        operator-(const Vector<_T, _l>& other)
        const
        {
            Vector<_T, _l> ret;
            #ifdef ARGON_SIMD_ENABLED
            vec_sub_simd(*this, other, ret);
            #else
            for (size_t i{}; i < _l; ++i)
                ret.items[i] = items[i] - other.items[i];
            #endif
            return ret;
        }

        inline Vector<_T, _l>
        operator*(const Vector<_T, _l>& other)
        const
        {
            Vector<_T, _l> ret;
            #ifdef ARGON_SIMD_ENABLED
            vec_mul_simd(*this, other, ret);
            #else
            for (size_t i{}; i < _l; ++i)
                ret.items[i] = items[i] * other.items[i];
            #endif
            return ret;
        }

        inline Vector<_T, _l>
        operator/(const Vector<_T, _l>& other)
        const
        {
            Vector<_T, _l> ret;
            #ifdef ARGON_SIMD_ENABLED
            vec_div_simd(*this, other, ret);
            #else
            for (size_t i{}; i < _l; ++i)
                ret.items[i] = items[i] / other.items[i];
            #endif
            return ret;
        }

        inline _T
        length()
        const
        {
            _T sum{0};
            #ifdef ARGON_SIMD_ENABLED
            vec_length_simd(*this, sum);
            #else
            for (size_t i{}; i < _l; ++i)
                sum += items[i] * items[i];
            #endif
            return (_T)sqrt(sum);
        }

        inline _T
        lengthSquared()
        const
        {
            _T sum{};
            #ifdef ARGON_SIMD_ENABLED
            vec_length_simd(*this, sum);
            #else
            for (size_t i{}; i < _l; ++i)
                sum += items[i] * items[i];
            #endif
            return sum;
        }

        inline void
        Normalize()
        const
        {
            _T len = this->length();
            #ifdef ARGON_SIMD_ENABLED
            vec_normalize_simd(*this, len);
            #else
            for(size_t i{}l i < _l; ++i)
                items[i] /= len;
            #endif
        }

        inline Vector<_T, _l>
        Normalized()
        const
        {
            _T len = this->length();
            Vector<_T, _l> ret = *this;
            #ifdef ARGON_SIMD_ENABLED
            vec_normalize_simd(ret, len);
            #else
            for (size_t i{}; i < _l; ++i)
                ret[i] /= len;
            #endif
            return ret;
        }

    };

    template <typename _T, size_t _l>
    inline std::ostream& 
    operator<<(std::ostream& stream, const Vector<_T, _l> &vec)
    {
        stream << "| ";
        for (size_t i{}; i < _l; ++i)
            stream << vec.items[i] << " ";
        stream << "|";
        return stream;
    }

    template <typename _T, size_t _l>
    inline _T
    Dot(const Vector<_T, _l> &right, const Vector<_T, _l> &left)
    {
        _T res{0};
        #ifdef ARGON_SIMD_ENABLED
        vec_dot_simd(right, left, res);
        #else
        for (size_t i{}; i < _l; ++i)
            res += right.items[i] * left[i];
        #endif
        return res;
    }

    template<typename _T>
    inline Vector<_T, 3>
    Cross(const Vector<_T, 3> &A, const Vector<_T, 3> &B)
    {
        Vector<_T, 3> ret;
        ret[0] = A[1] * B[2] - A[2] * B[1];
        ret[1] = A[2] * B[0] - A[0] * B[2];
        ret[2] = A[0] * B[1] - A[1] * B[0];
        return ret;
    }

    template <typename _T, size_t _l>
    inline _T
    Distance(const Vector<_T, _l> &A, const Vector<_T, _l> &B)
    {
        _T res{0};
        #ifdef ARGON_SIMD_ENABLED
        vec_distance_simd(A, B, res);
        #else
        for (size_t i{}; i < _l; ++i)
            res += (A[i] - B[i]) * (A[i] - B[i]);
        #endif
        return sqrt(res);
    }

    template<typename _T, size_t _l>
    inline _T
    DistanceSquared(const Vector<_T, _l> &A, const Vector<_T, _l> &B)
    {
        _T res{0};
        #ifdef ARGON_SIMD_ENABLED
        vec_distance_simd(A, B, res);
        #else
        for (size_t i{}; i < _l; ++i)
            res += (A[i] - B[i]) * (A[i] - B[i]);
        #endif
        return res;
    }

    template<typename _T, size_t _l>
    inline Vector<_T, _l>
    Lerp(const Vector<_T, _l> &A, const Vector<_T, _l> &B, _T t)
    {
        Vector<_T, _l> ret;
        #ifdef ARGON_SIMD_ENABLED
        vec_lerp_simd(A, B, t, ret);
        #else
        for (size_t i{}; i < _l; ++i)
            ret[i] = A[i] + (B[i] - A[i]) * t;
        #endif
        return ret;
    }

    template<typename _T, size_t _l>
    inline Vector<_T, _l>
    Reflect(const Vector<_T, _l> &vec, const Vector<_T, _l> &normal)
    {
        Vector<_T, _l> ret;
        #ifdef ARGON_SIMD_ENABLED
        vec_reflect_simd(vec, normal, ret);
        #else
        for (size_t i{}; i < _l; ++i)
            ret[i] = vec[i] - 2 * (vec[i] * normal[i]) * normal[i];
        #endif
        return ret;
    }

#ifdef ARGON_SIMD_ENABLED

    template<size_t _l>
    inline void
    vec_reflect_simd(const Vector<float, _l> &vec, const Vector<float, _l> &normal, Vector<float, _l> &ret)
    {
        simd::__m_float _vec = _mm_load_float(vec.items),
                        _nor = _mm_load_float(normal.items),
                        _2 = _mm_set1_ps(2),
                        _res;
        _res = _mm_mul_float(_vec, _nor); // (a * n)
        _res = _mm_mul_float(_res, _nor); // (a * n) * n
        _res = _mm_mul_float(_res, _2);   // 2 * (a * n) * n
        _res = _mm_sub_float(_vec, _res); // a - 2 * (a * n) * n
        _mm_store_float(ret.items, _res);
    }

    template <size_t _l>
    inline void
    vec_lerp_simd(const Vector<float, _l> &A, const Vector<float, _l> &B, float t, Vector<float, _l> &ret)
    {
        simd::__m_float _A = _mm_load_float(A.items),
                        _B = _mm_load_float(B.items),
                        _t = _mm_set1_ps(t),
                        _res;
        _res = _mm_sub_float(_B, _A);
        _res = _mm_add_float(_res, _A);
        _res = _mm_mul_float(_res, _t);
        _mm_store_float(ret.items, _res);
    }

    template<size_t _l>
    inline void
    vec_distance_simd(const Vector<float, _l> &A, const Vector<float, _l> &B, float &res)
    {
        simd::__m_float _A = _mm_load_float(A.items),
                        _B = _mm_load_float(B.items),
                        _res;
        _res = _mm_sub_float(_A, _B);
        _res = _mm_mul_float(_res, _res);
        _res = _mm_hadd_float(_res, _res);
        _res = _mm_hadd_float(_res, _res);
        res = _mm_first_float(_res);
    }

    template<size_t _l>
    inline void
    vec_normalize_simd(Vector<float, _l> &vec, const float len)
    {
        simd::__m_float _vec = _mm_load_float(vec.items),
                        _len = _mm_set1_ps(len);
        _vec = _mm_div_float(_vec, _len);
        _mm_store_float(vec.items, _vec);
    }

    template<size_t _l>
    inline void
    vec_dot_simd(const Vector<float, _l> &left, const Vector<float, _l> &right, float &ret)
    {
        simd::__m_float _A = _mm_load_float(left.items),
                        _B = _mm_load_float(right.items),
                        _res;
        _res = _mm_mul_float(_A, _B);
        _res = _mm_hadd_float(_res, _res);
        _res = _mm_hadd_float(_res, _res);
        ret = _mm_first_float(_res);
    }

    template<size_t _l>
    inline void
    vec_add_simd(const Vector<float, _l> &left, const Vector<float, _l> &right, Vector<float, _l> &ret)
    {
        simd::__m_float _A = _mm_load_float(left.items),
                        _B = _mm_load_float(right.items),
                        _res;
        _res = _mm_add_float(_A, _B);
        _mm_store_float(ret.items, _res);
    }

    template<size_t _l>
    inline void
    vec_sub_simd(const Vector<float, _l> &left, const Vector<float, _l> &right, Vector<float, _l> &ret)
    {
        simd::__m_float _A = _mm_load_float(left.items),
                        _B = _mm_load_float(right.items),
                        _res;
        _res = _mm_sub_float(_A, _B);
        _mm_store_float(ret.items, _res);
    }

    template<size_t _l>
    inline void
    vec_mul_simd(const Vector<float, _l> &left, const Vector<float, _l> &right, Vector<float, _l> &ret)
    {
        simd::__m_float _A = _mm_load_float(left.items),
                        _B = _mm_load_float(right.items),
                        _res;
        _res = _mm_mul_float(_A, _B);
        _mm_store_float(ret.items, _res);
    }

    template<size_t _l>
    inline void
    vec_div_simd(const Vector<float, _l> &left, const Vector<float, _l> &right, Vector<float, _l> &ret)
    {
        simd::__m_float _A = _mm_load_float(left.items),
                        _B = _mm_load_float(right.items),
                        _res;
        _res = _mm_div_float(_A, _B);
        _mm_store_float(ret.items, _res);
    }

    template<size_t _l>
    inline void
    vec_length_simd(const Vector<float, _l> &vec, float &ret)
    {
        simd::__m_float _A = _mm_load_float(vec.items);
        _A = _mm_mul_float(_A, _A);
        _A = _mm_hadd_float(_A, _A);
        _A = _mm_hadd_float(_A, _A);
        ret = _mm_first_float(_A);
    }

#endif



    using vec2 = Vector<float, 2>;
    using vec3 = Vector<float, 3>;
    using vec4 = Vector<float, 4>;

}

#endif