// Note and chord detector. C++ port of chord_detect.py (detect_notes / name_chord).
// No Arduino or ESP-IDF headers here: input is a float frame, output is plain data.
//
// Keep the constants in sync with the top of chord_detect.py.
#pragma once

#include <cstddef>

#include "fft.h"

namespace chord {

constexpr int kSampleRate = 16000;
constexpr int kFrameSize = kFftSize;  // N_FFT
constexpr int kHop = 2048;            // HOP
constexpr int kNoteLo = 48;           // NOTE_LO (C3)
constexpr int kNoteHi = 84;           // NOTE_HI (C6)
constexpr int kHarmonics = 6;         // N_HARM
constexpr int kMaxNotes = 6;          // MAX_NOTES
constexpr float kWeightPow = 0.5f;    // WEIGHT_POW
constexpr float kFundFrac = 0.15f;    // FUND_FRAC
constexpr float kNoteThreshold = 0.3f;  // NOTE_THRESHOLD
constexpr float kSilenceRms = 0.01f;    // SILENCE_RMS

class Detector {
public:
    Detector();

    // frame: kFrameSize samples in [-1, 1]. Writes up to kMaxNotes MIDI notes,
    // sorted ascending, into notes_out. Returns how many were found.
    int detect_notes(const float* frame, int* notes_out);

    // Hann window (symmetric, same as numpy.hanning), for testing.
    const float* window() const { return window_; }

private:
    float window_[kFrameSize];
    float windowed_[kFrameSize];
    float spec_[kFrameSize / 2 + 1];
    // bands_[note - kNoteLo][harmonic - 1] = {begin, end} bin range (end exclusive)
    short bands_[kNoteHi - kNoteLo + 1][kHarmonics][2];
};

// Names the chord formed by the notes (MIDI numbers), e.g. "C", "Am/C", "G7/B".
// Returns false if fewer than 3 notes or no known chord matches.
bool name_chord(const int* notes, int count, char* out, size_t out_size);

}  // namespace chord