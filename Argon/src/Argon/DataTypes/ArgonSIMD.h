#ifndef ARGON_SIMD
#define ARGON_SIMD

#if defined(__AVX512F__)
    #define ARGON_SIMD_ENABLED
    #define ARGON_SIMD_SIZE 512
#elif defined(__AVX2__)
    #define ARGON_SIMD_ENABLED
    #define ARGON_SIMD_SIZE 256
#elif defined(__AVX__)
    #define ARGON_SIMD_ENABLED
    #define ARGON_SIMD_SIZE 256
#elif defined(__SSE2__) || defined(__SSE__)
    #define ARGON_SIMD_ENABLED
    #define ARGON_SIMD_SIZE 128
#else
    #define ARGON_SIMD_DISABLED
    #define ARGON_SIMD_SIZE 0
#endif

// DEBUG ONLY
#define ARGON_SIMD_ENABLED
#define ARGON_SIMD_SIZE 256
// #define ARGON_SIMD_SIZE 128

#ifdef ARGON_SIMD_ENABLED
#include <immintrin.h>
#endif

/*
* -msse2 - SSE2
* -mavx - AVX
* -mavx2 - AVX2
* -mavx512f - AVX512
*/
namespace argon::simd
{
#if ARGON_SIMD_SIZE >= 256
    using __m_float = __m128;
    using __m_double = __m256d;

    constexpr size_t floatSteps = 4;
    constexpr size_t doubleSteps = 4;

    #define _mm_add_float _mm_add_ps
    #define _mm_sub_float _mm_sub_ps
    #define _mm_mul_float _mm_mul_ps
    #define _mm_div_float _mm_div_ps
    #define _mm_load_float _mm_load_ps
    #define _mm_store_float _mm_store_ps

    #define _mm_hadd_float _mm_hadd_ps
    #define _mm_first_float _mm_cvtss_f32

#elif ARGON_SIMD_SIZE >= 128
    using __m_float = __m128;
    using __m_double = __m128d;

    constexpr size_t floatSteps = 4;
    constexpr size_t doubleSteps = 2;

    #define _mm_add_float _mm_add_ps
    #define _mm_sub_float _mm_sub_ps
    #define _mm_mul_float _mm_mul_ps
    #define _mm_div_float _mm_div_ps
    #define _mm_load_float _mm_load_ps
    #define _mm_store_float _mm_store_ps

    #define _mm_hadd_float _mm_hadd_ps
    #define _mm_first_float _mm_cvtss_f32


#endif
    
#ifdef ARGON_SIMD_ENABLED
template <typename _T>
struct SimdAlignment
{
    static constexpr size_t value =
    sizeof(_T) == 4 ? 16 : 
    sizeof(_T) == 8 ? 32 : 
    alignof(_T);
};
#endif

}

#endif