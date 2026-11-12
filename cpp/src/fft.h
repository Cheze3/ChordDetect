// FFT backend interface. The detector only calls this one function, so the
// backend can be swapped: fft_simple.cpp on the Mac, an ESP-DSP version on the ESP32.
#pragma once

namespace chord {

constexpr int kFftSize = 4096;

// in:  kFftSize real samples (already windowed)
// out: kFftSize / 2 + 1 magnitudes |X[k]|, unscaled (same convention as numpy.fft.rfft)
void rfft_magnitude(const float* in, float* out);

}  // namespace chord