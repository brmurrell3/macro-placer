// k9.h: one kernel source for CUDA (NVRTC, nvcc) and HIP (hiprtc, hipcc). Prepended to every kernel file by k9cu.py.
//
// A kernel is written
//
//     K9_KERNEL name(float* out, const float* in, const int* ip, float fp0, int k9_n) {
//         K9_THREAD(i);            // i: this thread's index in [0, k9_n); threads that pad the last block return
//         ...
//     }
//
// The last parameter of every kernel is `int k9_n`, the number of threads the caller asked for (Metal dispatches
// exactly that many; a CUDA grid is rounded up to whole blocks). Where the Metal kernels took `constant float* fp`,
// the values come by value as float parameters (fp0, fp1, ...).
//
// The helpers below have the Metal Shading Language semantics of the functions they replace, so that a tie or a
// value on a bound is resolved as in the Metal kernels.
#if defined(__HIP__) || defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#define K9_HIP 1
#endif

#define K9_KERNEL extern "C" __global__ void
#define K9_THREAD(i) const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); if (i >= k9_n) return
#define K9_INLINE static __device__ __forceinline__

struct k9_f2 { float x, y; };

K9_INLINE float k9_min(float a, float b) { return b < a ? b : a; }          // MSL min(a, b)
K9_INLINE float k9_max(float a, float b) { return a < b ? b : a; }          // MSL max(a, b)
K9_INLINE int k9_mini(int a, int b) { return b < a ? b : a; }
K9_INLINE int k9_maxi(int a, int b) { return a < b ? b : a; }
K9_INLINE float k9_clamp(float x, float lo, float hi) { return k9_min(k9_max(x, lo), hi); }   // MSL clamp
K9_INLINE int k9_clampi(int x, int lo, int hi) { return k9_mini(k9_maxi(x, lo), hi); }
K9_INLINE float k9_step(float edge, float x) { return x < edge ? 0.0f : 1.0f; }               // MSL step
K9_INLINE float k9_sign(float x) { return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f); }      // MSL sign
// number of trailing zero bits of a non-zero word (MSL ctz)
K9_INLINE int k9_ctz(unsigned int x) {
#ifdef K9_HIP
    return __builtin_ctz(x);
#else
    return __ffs((int)x) - 1;
#endif
}

// float32 sum and its rounding error (no multiplication, so contraction of multiply-add cannot touch it)
K9_INLINE k9_f2 k9_two_sum(float a, float b) {
    float s = a + b;
    float bb = s - a;
    k9_f2 r; r.x = s; r.y = (a - (s - bb)) + (b - bb);
    return r;
}
