#ifndef ARGON_VECTOR
#define ARGON_VECTOR

#include "ArgonSIMD.h"

#include <cstddef>
#include <type_traits>
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
        inline
        Vector() 
        {
            for (size_t i{}; i < SIMDPadding; ++i)
                items[i] = _T{0};
        }
        inline
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

    };

    template <typename _T, size_t _l>
    inline std::ostream& 
    operator<<(std::ostream& stream, const Vector<_T, _l>& vec)
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

#ifdef ARGON_SIMD_ENABLED

    template<size_t _l>
    inline void
    vec_dot_simd(const Vector<float, _l>& left, const Vector<float, _l>& right, float& ret)
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
    vec_dot_simd(const Vector<double, _l>& left, const Vector<double, _l>& right, double& ret)
    {
        simd::__m_double _A = _mm_load_double(left.items),
                         _B = _mm_load_double(right.items),
                         _res;
        #if ARGON_SIMD_SIZE >= 256
        _res = _mm_mul_double(_A, _B);
        _res = _mm_hadd_double(_res, _res);
        _res = _mm256_shuffle_pd(_res, _res, _MM_SHUFFLE(0, 2, 0, 2));
        _res = _mm_hadd_double(_res, _res);
        __m128d _lower = _mm_half_double(_res);
        ret = _mm_first_double(_lower);
        #else
        _res = _mm_mul_double(_A, _B);
        _res = _mm_hadd_double(_res, _res);
        ret = _mm_first_double(_res);
        _A = _mm_load_double(&left.items[2]);
        _B = _mm_load_double(&right.items[2]);
        _res = _mm_mul_double(_A, _B);
        _res = _mm_hadd_double(_res, _res);
        ret += _mm_first_double(_res);
        #endif
    }

    template<size_t _l>
    inline void
    vec_add_simd(const Vector<float, _l>& left, const Vector<float, _l>& right, Vector<float, _l>& ret)
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
    vec_add_simd(const Vector<double, _l> &left, const Vector<double, _l> &right, Vector<double, _l> &ret)
    {
        simd::__m_double _A = _mm_load_double(left.items),
                         _B = _mm_load_double(right.items),
                         _res;
    #if ARGON_SIMD_SIZE >= 256 // AVX
        _res = _mm_add_double(_A, _B);
        _mm_store_double(ret.items, _res);
    #else // SSE
        _res = _mm_add_double(_A, _B);
        _mm_store_double(ret.items, _res);
        _A = _mm_load_double(&left.items[2]);
        _B = _mm_load_double(&right.items[2]);
        _res = _mm_add_double(_A, _B);
        _mm_store_double(&ret.items[2], _res);
    #endif
    }

    template <size_t _l>
    inline void
    vec_sub_simd(const Vector<double, _l> &left, const Vector<double, _l> &right, Vector<double, _l> &ret)
    {
        simd::__m_double _A = _mm_load_double(left.items),
                         _B = _mm_load_double(right.items),
                         _res;
    #if ARGON_SIMD_SIZE >= 256 // AVX
        _res = _mm_sub_double(_A, _B);
        _mm_store_double(ret.items, _res);
    #else // SSE
        _res = _mm_sub_double(_A, _B);
        _mm_store_double(ret.items, _res);
        _A = _mm_load_double(&left.items[2]);
        _B = _mm_load_double(&right.items[2]);
        _res = _mm_sub_double(_A, _B);
        _mm_store_double(&ret.items[2], _res);
    #endif
    }

    template <size_t _l>
    inline void
    vec_mul_simd(const Vector<double, _l> &left, const Vector<double, _l> &right, Vector<double, _l> &ret)
    {
        simd::__m_double _A = _mm_load_double(left.items),
                         _B = _mm_load_double(right.items),
                         _res;
    #if ARGON_SIMD_SIZE >= 256 // AVX
        _res = _mm_mul_double(_A, _B);
        _mm_store_double(ret.items, _res);
    #else // SSE
        _res = _mm_mul_double(_A, _B);
        _mm_store_double(ret.items, _res);
        _A = _mm_load_double(&left.items[2]);
        _B = _mm_load_double(&right.items[2]);
        _res = _mm_mul_double(_A, _B);
        _mm_store_double(&ret.items[2], _res);
    #endif
    }

    template <size_t _l>
    inline void
    vec_div_simd(const Vector<double, _l> &left, const Vector<double, _l> &right, Vector<double, _l> &ret)
    {
        simd::__m_double _A = _mm_load_double(left.items),
                         _B = _mm_load_double(right.items),
                         _res;
    #if ARGON_SIMD_SIZE >= 256 // AVX
        _res = _mm_div_double(_A, _B);
        _mm_store_double(ret.items, _res);
    #else // SSE
        _res = _mm_div_double(_A, _B);
        _mm_store_double(ret.items, _res);
        _A = _mm_load_double(&left.items[2]);
        _B = _mm_load_double(&right.items[2]);
        _res = _mm_div_double(_A, _B);
        _mm_store_double(&ret.items[2], _res);
    #endif
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

    template<size_t _l>
    inline void
    vec_length_simd(const Vector<double, _l> &vec, double &ret)
    {
        simd::__m_double _A = _mm_load_double(vec.items);
        _A = _mm_mul_double(_A, _A);
        _A = _mm_hadd_double(_A, _A);
        #if ARGON_SIMD_SIZE >= 256
        _A = _mm_hadd_double(_A, _A);
        __m128d _lower = _mm_half_double(_A);
        ret = _mm_first_double(_lower);
        #else
        ret = _mm_first_double(_A);
        _A = _mm_load_double(&vec.items[2]);
        _A = _mm_mul_double(_A, _A);
        _A = _mm_hadd_double(_A, _A);
        ret += _mm_first_double(_A);
        #endif
    }

#endif



    using vec2f = Vector<float, 2>;
    using vec3f = Vector<float, 3>;
    using vec4f = Vector<float, 4>;
    using vec2d = Vector<double, 2>;
    using vec3d = Vector<double, 3>;
    using vec4d = Vector<double, 4>;

}

#endif