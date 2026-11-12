#include "detector.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace chord {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kBins = kFrameSize / 2 + 1;

double midi_to_hz(int m) { return 440.0 * std::pow(2.0, (m - 69) / 12.0); }

const char* const kNoteNames[12] = {"C",  "C#", "D",  "D#", "E",  "F",
                                    "F#", "G",  "G#", "A",  "A#", "B"};

struct ChordType {
    const char* suffix;
    int count;
    int intervals[4];
};

// Same order as CHORD_TYPES in chord_detect.py (order matters for ties).
const ChordType kChordTypes[] = {
    {"", 3, {0, 4, 7, 0}},      {"m", 3, {0, 3, 7, 0}},
    {"dim", 3, {0, 3, 6, 0}},   {"aug", 3, {0, 4, 8, 0}},
    {"sus2", 3, {0, 2, 7, 0}},  {"sus4", 3, {0, 5, 7, 0}},
    {"maj7", 4, {0, 4, 7, 11}}, {"7", 4, {0, 4, 7, 10}},
    {"m7", 4, {0, 3, 7, 10}},
};

}  // namespace

Detector::Detector() {
    // np.hanning(N): 0.5 - 0.5 * cos(2*pi*n / (N-1))
    for (int i = 0; i < kFrameSize; ++i) {
        window_[i] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * i / (kFrameSize - 1)));
    }
    const double df = static_cast<double>(kSampleRate) / kFrameSize;
    for (int m = kNoteLo; m <= kNoteHi; ++m) {
        for (int h = 1; h <= kHarmonics; ++h) {
            // Python round() is ties-to-even, which matches nearbyint in the default mode.
            const int c = static_cast<int>(std::nearbyint(midi_to_hz(m) * h / df));
            bands_[m - kNoteLo][h - 1][0] = static_cast<short>(std::max(c - 1, 0));
            bands_[m - kNoteLo][h - 1][1] = static_cast<short>(std::min(c + 2, kBins));
        }
    }
}

int Detector::detect_notes(const float* frame, int* notes_out) {
    double sum_sq = 0.0;
    for (int i = 0; i < kFrameSize; ++i) sum_sq += static_cast<double>(frame[i]) * frame[i];
    if (std::sqrt(sum_sq / kFrameSize) < kSilenceRms) return 0;

    for (int i = 0; i < kFrameSize; ++i) windowed_[i] = frame[i] * window_[i];
    rfft_magnitude(windowed_, spec_);

    float smax = 0.0f;
    for (int k = 0; k < kBins; ++k) smax = std::max(smax, spec_[k]);

    auto band_max = [this](int m, int h) {
        const short* b = bands_[m - kNoteLo][h - 1];
        float v = spec_[b[0]];
        for (int k = b[0] + 1; k < b[1]; ++k) v = std::max(v, spec_[k]);
        return v;
    };

    int found[kMaxNotes];
    int n_found = 0;
    float first = 0.0f;
    bool have_first = false;

    for (int iter = 0; iter < kMaxNotes; ++iter) {
        int best_m = -1;
        float best_s = 0.0f;
        for (int m = kNoteLo; m <= kNoteHi; ++m) {
            bool taken = false;
            for (int i = 0; i < n_found; ++i) taken = taken || (found[i] == m);
            if (taken) continue;
            if (band_max(m, 1) < kFundFrac * smax) continue;
            float s = 0.0f;
            for (int h = 1; h <= kHarmonics; ++h) {
                s += band_max(m, h) * std::pow(static_cast<float>(h), -kWeightPow);
            }
            if (s > best_s) {
                best_m = m;
                best_s = s;
            }
        }
        if (best_m < 0) break;
        if (!have_first) {
            first = best_s;
            have_first = true;
        } else if (best_s < kNoteThreshold * first) {
            break;
        }
        found[n_found++] = best_m;

        const float p1 = band_max(best_m, 1);
        for (int h = 1; h <= kHarmonics; ++h) {
            const short* b = bands_[best_m - kNoteLo][h - 1];
            for (int k = b[0]; k < b[1]; ++k) {
                spec_[k] = std::max(spec_[k] - p1 / h, 0.0f);
            }
        }
    }

    // Insertion sort (at most kMaxNotes elements).
    for (int i = 1; i < n_found; ++i) {
        const int v = found[i];
        int j = i - 1;
        while (j >= 0 && found[j] > v) {
            found[j + 1] = found[j];
            --j;
        }
        found[j + 1] = v;
    }
    for (int i = 0; i < n_found; ++i) notes_out[i] = found[i];
    return n_found;
}

bool name_chord(const int* notes, int count, char* out, size_t out_size) {
    if (count < 3) return false;
    unsigned pcs = 0;
    int lowest = notes[0];
    for (int i = 0; i < count; ++i) {
        pcs |= 1u << (notes[i] % 12);
        lowest = std::min(lowest, notes[i]);
    }
    const int bass = lowest % 12;

    int first_root = -1, first_type = -1;      // first match overall
    int bass_root = -1, bass_type = -1;        // first match whose root is the bass
    for (int root = 0; root < 12; ++root) {
        for (int t = 0; t < static_cast<int>(sizeof(kChordTypes) / sizeof(kChordTypes[0])); ++t) {
            unsigned mask = 0;
            for (int i = 0; i < kChordTypes[t].count; ++i) {
                mask |= 1u << ((root + kChordTypes[t].intervals[i]) % 12);
            }
            if (mask != pcs) continue;
            if (first_root < 0) {
                first_root = root;
                first_type = t;
            }
            if (root == bass && bass_root < 0) {
                bass_root = root;
                bass_type = t;
            }
        }
    }
    if (first_root < 0) return false;
    const int root = bass_root >= 0 ? bass_root : first_root;
    const int type = bass_root >= 0 ? bass_type : first_type;
    if (root == bass) {
        std::snprintf(out, out_size, "%s%s", kNoteNames[root], kChordTypes[type].suffix);
    } else {
        std::snprintf(out, out_size, "%s%s/%s", kNoteNames[root], kChordTypes[type].suffix,
                      kNoteNames[bass]);
    }
    return true;
}

}  // namespace chord