//===- blur3x3.cc -------------------------------------------------*- C++ -*-===//
// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Scott Boudreaux / Elyan Labs. Commercial license: see ../../COMMERCIAL.md
//
// Custom AIE2 3x3 blur kernel for the XDNA NPU. Unlike mlir-aie's stock filter2d (whose VECTOR
// path truncates the int16 kernel to int8 via >>8 and saturates bright unity-gain outputs to the
// int8-positive range — fine for the zero-centered Laplacian, broken for a blur), this uses an
// int32 accumulator with the int16 kernel directly, >>SRS_SHIFT, and a proper [0,255] uint8 clamp.
// For a unity-sum kernel (e.g. Gaussian [[256,512,256],[512,1024,512],[256,512,256]] summing 4096):
// uniform N -> N*4096 >> 12 = N. Exact unity gain, correctly saturated.
//
// Symbol: blurLine(l0,l1,l2,out,width,kernel) — same ABI as filter2d's filter2dLine, so it drops
// into the same 3-line-stencil worker. (Scalar path: correctness over peak throughput; the camera
// pipeline has large FPS headroom.)

#define NOCPP
#include <stdint.h>
#include <aie_api/aie.hpp>

const int32_t SRS_SHIFT = 12;

static inline void blur3x3_scalar(uint8_t *lineIn0, uint8_t *lineIn1,
                                  uint8_t *lineIn2, uint8_t *output,
                                  const int32_t width, int16_t *kernel) {
  int32_t acc;

  // left column: border extension by replicating column 0
  acc = 0;
  acc += ((int32_t)lineIn0[0]) * kernel[0 * 3 + 0];
  acc += ((int32_t)lineIn1[0]) * kernel[1 * 3 + 0];
  acc += ((int32_t)lineIn2[0]) * kernel[2 * 3 + 0];
  for (int ki = 1; ki < 3; ki++) {
    acc += ((int32_t)lineIn0[0 + ki - 1]) * kernel[0 * 3 + ki];
    acc += ((int32_t)lineIn1[0 + ki - 1]) * kernel[1 * 3 + ki];
    acc += ((int32_t)lineIn2[0 + ki - 1]) * kernel[2 * 3 + ki];
  }
  acc = ((acc + (1 << (SRS_SHIFT - 1))) >> SRS_SHIFT);
  acc = (acc > UINT8_MAX) ? UINT8_MAX : (acc < 0) ? 0 : acc;
  output[0] = (uint8_t)acc;

  // interior
  for (int i = 1; i < width - 1; i++) {
    acc = 0;
    for (int ki = 0; ki < 3; ki++) {
      acc += ((int32_t)lineIn0[i + ki - 1]) * kernel[0 * 3 + ki];
      acc += ((int32_t)lineIn1[i + ki - 1]) * kernel[1 * 3 + ki];
      acc += ((int32_t)lineIn2[i + ki - 1]) * kernel[2 * 3 + ki];
    }
    acc = ((acc + (1 << (SRS_SHIFT - 1))) >> SRS_SHIFT);
    acc = (acc > UINT8_MAX) ? UINT8_MAX : (acc < 0) ? 0 : acc;
    output[i] = (uint8_t)acc;
  }

  // right column: border extension by replicating last column
  acc = 0;
  for (int ki = 0; ki < 2; ki++) {
    acc += ((int32_t)lineIn0[width + ki - 2]) * kernel[0 * 3 + ki];
    acc += ((int32_t)lineIn1[width + ki - 2]) * kernel[1 * 3 + ki];
    acc += ((int32_t)lineIn2[width + ki - 2]) * kernel[2 * 3 + ki];
  }
  acc += ((int32_t)lineIn0[width - 1]) * kernel[0 * 3 + 2];
  acc += ((int32_t)lineIn1[width - 1]) * kernel[1 * 3 + 2];
  acc += ((int32_t)lineIn2[width - 1]) * kernel[2 * 3 + 2];
  acc = ((acc + (1 << (SRS_SHIFT - 1))) >> SRS_SHIFT);
  acc = (acc > UINT8_MAX) ? UINT8_MAX : (acc < 0) ? 0 : acc;
  output[width - 1] = (uint8_t)acc;
}

extern "C" {
void blurLine(uint8_t *lineIn0, uint8_t *lineIn1, uint8_t *lineIn2,
              uint8_t *out, int32_t lineWidth, int16_t *blurKernel) {
  blur3x3_scalar(lineIn0, lineIn1, lineIn2, out, lineWidth, blurKernel);
}
} // extern "C"
