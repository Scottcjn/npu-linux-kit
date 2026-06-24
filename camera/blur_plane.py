# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Scott Boudreaux / Elyan Labs. Commercial license: see ../COMMERCIAL.md
#
# Purpose-built NPU blur kernel: a CLEAN single-channel 3x3 Gaussian filter2d on the AIE.
# No rgba2gray round-trip, no add_weighted overlay (the speckle + 2x-gain sources). The host
# feeds each color channel as a uint8 plane (multi-pass = re-feed) and recombines -> clean color blur.
#
# Dataflow FIX 2026-06-24: the filter's 3-line stencil consumes a WORKER-PRODUCED line stream
# (a passthrough copy producer), exactly like edge_detect's verified rgba2gray->filter structure —
# NOT a raw DMA forward (which caused value-dependent gain + line streaks).
#
# Unity gain: int16 coeffs >>8 -> int8 [[1,2,1],[2,4,2],[1,2,1]] (sum 16); filter2d stores
# acc >> (SRS_SHIFT-8 = 4) = acc/16 = exact weighted mean.

import sys, os
#
# DEFINITIVE DIAGNOSIS 2026-06-24: with the correct (worker-produced) dataflow AND unity-gain coeffs,
# the stock kernels.filter2d STILL produces value-dependent output that SATURATES bright pixels to the
# int8-positive range: uniform 100->100 but 200->132, 255->150. The kernel's accumulation/store path
# is built for the Laplacian (zero-centered edge outputs) and clamps a unity-sum blur of bright images.
# This is NOT a dataflow or coeff bug (both fixed). CONCLUSION: a clean NPU blur needs a CUSTOM AIE
# blur kernel (.cc, hand-written vector intrinsics with a proper uint8 saturate-store), not filter2d.
# Until then the kit's background blur uses the clean CPU Gaussian (camera/README).
import numpy as np
import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker, kernels
from aie.iron.controlflow import range_
from aie.iron.kernels._common import _make_extern


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def blur_plane(
    in_tensor: In,
    out_tensor: Out,
    *,
    width: CompileTime[int] = 1280,
    height: CompileTime[int] = 720,
):
    height_minus_1 = height - 1
    line_width = width
    line_ty = np.ndarray[(line_width,), np.dtype[np.uint8]]

    pass_through_line = kernels.passthrough(tile_size=line_width, dtype=np.uint8)
    _kernel_ty = np.ndarray[(3, 3), np.dtype[np.int16]]
    _blur_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "blur3x3.cc")
    filter2d_line_kernel = _make_extern(
        "blurLine", _blur_src,
        [line_ty, line_ty, line_ty, line_ty, np.int32, _kernel_ty],
    )
    # unity-gain Gaussian (>>8 -> int8 [1,2,1;2,4,2;1,2,1], sum 16; filter2d >>4 = /16)
    filter_kernel_buff = Buffer(
        np.ndarray[(3, 3), np.dtype[np.int16]],
        name="kernel",
        initial_value=np.array(
            [[256, 512, 256], [512, 1024, 512], [256, 512, 256]], dtype=np.int16
        ),
    )

    of_in = ObjectFifo(line_ty, name="in")                       # DMA input
    of_lines = ObjectFifo(line_ty, depth=4, name="lines")        # worker-produced line stream (filter input)
    of_out = ObjectFifo(line_ty, name="out")                     # DMA output

    workers = []

    # producer: copy DMA input lines into of_lines, one per call (IRON auto-iterates over height)
    def copy_fn(of_i, of_o, k):
        ein = of_i.acquire(1)
        eout = of_o.acquire(1)
        k(ein, eout, line_width)
        of_i.release(1)
        of_o.release(1)

    workers.append(Worker(copy_fn, [of_in.cons(), of_lines.prod(), pass_through_line]))

    # filter: 3-line stencil over the produced line stream
    def filter_fn(of_i, of_o, kernel, f2d):
        for _ in range_(sys.maxsize):
            ein = of_i.acquire(2)                                # top border: dup first row
            eout = of_o.acquire(1)
            f2d(ein[0], ein[0], ein[1], eout, line_width, kernel)
            of_o.release(1)
            for _ in range_(1, height_minus_1):                  # steady (i-1, i, i+1)
                ein = of_i.acquire(3)
                eout = of_o.acquire(1)
                f2d(ein[0], ein[1], ein[2], eout, line_width, kernel)
                of_i.release(1)
                of_o.release(1)
            ein = of_i.acquire(2)                                # bottom border: dup last row
            eout = of_o.acquire(1)
            f2d(ein[0], ein[1], ein[1], eout, line_width, kernel)
            of_i.release(2)
            of_o.release(1)

    workers.append(
        Worker(filter_fn, [of_lines.cons(), of_out.prod(), filter_kernel_buff, filter2d_line_kernel],
               while_true=False)
    )

    plane_ty = np.ndarray[(width * height,), np.dtype[np.uint8]]
    rt = Runtime()
    with rt.sequence(plane_ty, plane_ty) as (i_in, o_out):
        rt.start(*workers)
        rt.fill(of_in.prod(), i_in)
        rt.drain(of_out.cons(), o_out, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()
