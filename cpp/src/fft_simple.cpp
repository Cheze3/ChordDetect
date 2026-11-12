// Self-contained radix-2 FFT for desktop testing. No dependencies.
// It treats the real input as complex (zero imaginary part), which is about
// twice as slow as a real-input FFT but irrelevant on a Mac.
#include <cmath>

#include "fft.h"

namespace chord {

namespace {

constexpr int N = kFftSize;
constexpr double kPi = 3.14159265358979323846;

struct Tables {
    float cos_t[N / 2];
    float sin_t[N / 2];
    unsigned short rev[N];
    Tables() {
        for (int i = 0; i < N / 2; ++i) {
            cos_t[i] = static_cast<float>(std::cos(2.0 * kPi * i / N));
            sin_t[i] = static_cast<float>(-std::sin(2.0 * kPi * i / N));
        }
        int bits = 0;
        while ((1 << bits) < N) ++bits;
        for (int i = 0; i < N; ++i) {
            int r = 0;
            for (int b = 0; b < bits; ++b) {
                if (i & (1 << b)) r |= 1 << (bits - 1 - b);
            }
            rev[i] = static_cast<unsigned short>(r);
        }
    }
};

}  // namespace

void rfft_magnitude(const float* in, float* out) {
    static const Tables t;
    static float re[N];
    static float im[N];

    for (int i = 0; i < N; ++i) {
        re[t.rev[i]] = in[i];
        im[t.rev[i]] = 0.0f;
    }
    for (int len = 2; len <= N; len <<= 1) {
        const int half = len / 2;
        const int step = N / len;
        for (int start = 0; start < N; start += len) {
            for (int k = 0; k < half; ++k) {
                const float wr = t.cos_t[k * step];
                const float wi = t.sin_t[k * step];
                const int a = start + k;
                const int b = a + half;
                const float tr = re[b] * wr - im[b] * wi;
                const float ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
    }
    for (int k = 0; k <= N / 2; ++k) {
        out[k] = std::sqrt(re[k] * re[k] + im[k] * im[k]);
    }
}

}  // namespace chord