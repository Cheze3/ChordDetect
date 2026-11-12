// Compares the C++ detector against vectors exported by export_vectors.py.
// Checks run in layers so a failure points at the stage that broke:
//   1. window   2. FFT   3. chord naming   4. note detection (+ chord from those notes)
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "detector.h"

namespace {

using chord::kFrameSize;

std::vector<float> read_floats(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr, "cannot open %s (run export_vectors.py first)\n", path.c_str());
        std::exit(2);
    }
    const std::streamsize bytes = f.tellg();
    f.seekg(0);
    std::vector<float> v(static_cast<size_t>(bytes) / sizeof(float));
    f.read(reinterpret_cast<char*>(v.data()), bytes);
    return v;
}

std::string join(const std::vector<int>& v) {
    std::string s;
    for (int x : v) s += (s.empty() ? "" : " ") + std::to_string(x);
    return s.empty() ? "(none)" : s;
}

std::string chord_of(const std::vector<int>& notes) {
    char buf[32];
    return chord::name_chord(notes.data(), static_cast<int>(notes.size()), buf, sizeof buf)
               ? std::string(buf)
               : std::string("-");
}

bool report(const char* stage, int failures, int total) {
    std::printf("%-22s %d/%d passed\n", stage, total - failures, total);
    return failures == 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "test_data";
    bool ok = true;
    chord::Detector det;

    // 1. Window
    const std::vector<float> py_window = read_floats(dir + "/window.bin");
    double wmax = 0;
    for (int i = 0; i < kFrameSize; ++i) wmax = std::fmax(wmax, std::fabs(det.window()[i] - py_window[i]));
    std::printf("window max abs diff    %.2e\n", wmax);
    ok &= wmax < 1e-6;

    // 2. FFT, using Python's window so only the FFT is under test
    const std::vector<float> frames = read_floats(dir + "/frames.bin");
    const std::vector<float> spectra = read_floats(dir + "/spectra.bin");
    const int bins = kFrameSize / 2 + 1;
    const int n_spectra = static_cast<int>(spectra.size()) / bins;
    double worst_rel = 0;
    std::vector<float> windowed(kFrameSize), mag(bins);
    for (int s = 0; s < n_spectra; ++s) {
        for (int i = 0; i < kFrameSize; ++i) windowed[i] = frames[s * kFrameSize + i] * py_window[i];
        chord::rfft_magnitude(windowed.data(), mag.data());
        double peak = 0, diff = 0;
        for (int k = 0; k < bins; ++k) {
            peak = std::fmax(peak, spectra[s * bins + k]);
            diff = std::fmax(diff, std::fabs(mag[k] - spectra[s * bins + k]));
        }
        worst_rel = std::fmax(worst_rel, peak > 0 ? diff / peak : diff);
    }
    std::printf("fft max error/peak     %.2e (%d spectra)\n", worst_rel, n_spectra);
    ok &= worst_rel < 1e-4;

    // 3. Chord naming from given notes
    {
        std::ifstream f(dir + "/naming.txt");
        std::string line;
        int total = 0, fail = 0;
        while (std::getline(f, line)) {
            std::istringstream in(line);
            int n;
            in >> n;
            std::vector<int> notes(n);
            for (int& x : notes) in >> x;
            std::string expected;
            in >> expected;
            ++total;
            const std::string got = chord_of(notes);
            if (got != expected) {
                if (++fail <= 10) std::printf("  naming: [%s] expected %s got %s\n", join(notes).c_str(), expected.c_str(), got.c_str());
            }
        }
        ok &= report("chord naming", fail, total);
    }

    // 4. Note detection and chord from the detected notes
    {
        std::ifstream f(dir + "/expected.txt");
        std::string line;
        int total = 0, note_fail = 0, chord_fail = 0;
        while (std::getline(f, line)) {
            std::istringstream in(line);
            int idx, n;
            std::string kind;
            in >> idx >> kind >> n;
            std::vector<int> expected(n);
            for (int& x : expected) in >> x;
            std::string expected_chord;
            in >> expected_chord;

            int out[chord::kMaxNotes];
            const int got_n = det.detect_notes(&frames[static_cast<size_t>(idx) * kFrameSize], out);
            std::vector<int> got(out, out + got_n);
            ++total;
            if (got != expected) {
                if (++note_fail <= 10) std::printf("  notes #%d %s: expected [%s] got [%s]\n", idx, kind.c_str(), join(expected).c_str(), join(got).c_str());
            } else if (chord_of(got) != expected_chord) {
                ++chord_fail;
            }
        }
        ok &= report("note detection", note_fail, total);
        ok &= report("chord from notes", chord_fail, total - note_fail);
    }

    std::printf("\n%s\n", ok ? "ALL PASS" : "FAILURES");
    return ok ? 0 : 1;
}
