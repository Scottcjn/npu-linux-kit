# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Scott Boudreaux / Elyan Labs. Commercial license: see ../COMMERCIAL.md
#
# Purpose-built NPU blur kernel: a CLEAN single-channel 3x3 Gaussian filter2d on the AIE.
# Unlike the edge-detect-adapted blur, this does ONLY filter2d on a raw uint8 plane — no
# rgba2gray round-trip, no add_weighted overlay (the sources of speckle + 2x gain). The host
# feeds each color channel as a plane (multi-pass = re-feed) and recombines → clean color blur.
#
# Unity gain: int16 coeffs >>8 -> int8 [[1,2,1],[2,4,2],[1,2,1]] (sum 16); filter2d stores
# acc >> (SRS_SHIFT-8 = 4) = acc/16 = exact weighted mean.

import sys
#
# WIP STATUS 2026-06-24: right architecture (clean single-plane, no rgba2gray/add_weighted), and the
# filter2d kernel math is sound (uint8 data, int8 coeffs >>8, acc>>4 = unity). BUT this hand-wired
# 3-line-stencil dataflow still shows a value-dependent gain (uniform 64->64 ok, 128->86, 200->129)
# + horizontal line streaks -> the stencil isn't consistently fed correct lines (fifo/forward wiring).
# NEXT: rebase the stencil on mlir-aie's verified conv2d/filter programming_example instead of
# adapting edge_detect's worker graph. Until then, the kit's background blur uses the clean CPU path.
import numpy as np
import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker, kernels
from aie.iron.controlflow import range_


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

    filter2d_line_kernel = kernels.filter2d(line_width=line_width)
    # unity-gain Gaussian (>>8 -> int8 [1,2,1;2,4,2;1,2,1], sum 16; filter2d >>4 = /16)
    filter_kernel_buff = Buffer(
        np.ndarray[(3, 3), np.dtype[np.int16]],
        name="kernel",
        initial_value=np.array(
            [[256, 512, 256], [512, 1024, 512], [256, 512, 256]], dtype=np.int16
        ),
    )

    in_of = ObjectFifo(line_ty, name="inOF")
    in_l1 = in_of.cons(4).forward(depth=4, name="inOF_L1")
    out_of = ObjectFifo(line_ty, name="outOF")
    out_l3 = out_of.cons().forward(name="outOF_L3")

    def filter_fn(of_in, of_out, filter_kernel, filter2d_line):
        for _ in range_(sys.maxsize):
            # top border: duplicate first row
            ein = of_in.acquire(2)
            eout = of_out.acquire(1)
            filter2d_line(ein[0], ein[0], ein[1], eout, line_width, filter_kernel)
            of_out.release(1)
            # steady state: rows (i-1, i, i+1)
            for _ in range_(1, height_minus_1):
                ein = of_in.acquire(3)
                eout = of_out.acquire(1)
                filter2d_line(ein[0], ein[1], ein[2], eout, line_width, filter_kernel)
                of_in.release(1)
                of_out.release(1)
            # bottom border: duplicate last row
            ein = of_in.acquire(2)
            eout = of_out.acquire(1)
            filter2d_line(ein[0], ein[1], ein[1], eout, line_width, filter_kernel)
            of_in.release(2)
            of_out.release(1)

    worker = Worker(
        filter_fn,
        [in_l1.cons(), out_of.prod(), filter_kernel_buff, filter2d_line_kernel],
        while_true=False,
    )

    plane_ty = np.ndarray[(width * height,), np.dtype[np.uint8]]
    rt = Runtime()
    with rt.sequence(plane_ty, plane_ty) as (i_in, o_out):
        rt.start(worker)
        rt.fill(in_of.prod(), i_in)
        rt.drain(out_l3.cons(), o_out, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()
