from math import gamma

import numpy as np

from edgfs2D.std.scattering.base import BaseScatteringModel


# Simplified VHS model for GLL based nodal collocation schemes
class FastSpectral(BaseScatteringModel):
    kind = "fast-spectral-vhs"

    def __init__(self, cfg, sect, velocitymesh, *args, **kwargs):
        super().__init__(cfg, sect, velocitymesh, *args, **kwargs)
        self.load_parameters()
        # self.perform_precomputation()
        print("scattering-model: finished computation")

    def load_parameters(self):
        sect = self.sect
        nd = self.nondim
        alpha = 1.0
        omega = self.cfg.lookupfloat(sect, "omega")
        self._gamma = 2.0 * (1 - omega)

        dRef = self.cfg.lookupfloat(sect, "dRef")
        Tref = self.cfg.lookupfloat(sect, "Tref")

        invKn = (
            nd.H0
            * np.sqrt(2.0)
            * np.pi
            * nd.n0
            * dRef
            * dRef
            * pow(Tref / nd.T0, omega - 0.5)
        )

        self._prefactor = (
            invKn * alpha / (pow(2.0, 2 - omega + alpha) * gamma(2.5 - omega) * np.pi)
        )
        self._omega = omega

    def perform_precomputation(self):
        # Precompute aa, bb1, bb2 (required for kernel)
        # compute l
        N = self.vm.Nv()
        Nrho = self.vm.Nrho()
        M = self.vm.M()
        L = self.vm.L()
        qz = self.vm.qz()
        qw = self.vm.qw()
        sz = self.vm.sz()
        vsize = self.vm.vsize()

        l0 = np.concatenate((np.arange(0, N / 2), np.arange(-N / 2, 0)))
        # l = l0[np.mgrid[0:N, 0:N, 0:N]]
        # l = l.reshape((3,vsize))
        l = np.zeros((3, vsize))
        for idv in range(vsize):
            I = int(idv / (N * N))
            J = int((idv % (N * N)) / N)
            K = int((idv % (N * N)) % N)
            l[0, idv] = l0[I]
            l[1, idv] = l0[J]
            l[2, idv] = l0[K]
        d_lx = gpuarray.to_gpu(np.ascontiguousarray(l[0, :]))
        d_ly = gpuarray.to_gpu(np.ascontiguousarray(l[1, :]))
        d_lz = gpuarray.to_gpu(np.ascontiguousarray(l[2, :]))

        dtype = self.cfg.dtype
        cdtype = np.complex128
        CUFFT_T2T = CUFFT_Z2Z
        self.cufftExecT2T = cufftExecZ2Z

        if dtype == np.float32:
            cdtype = np.complex64
            CUFFT_T2T = CUFFT_C2C
            self.cufftExecT2T = cufftExecC2C

        # define scratch  spaces
        self.d_FTf = gpuarray.empty(vsize, dtype=cdtype)
        self.d_fC = gpuarray.empty_like(self.d_FTf)
        self.d_QG = gpuarray.empty_like(self.d_FTf)
        self.d_t1 = gpuarray.empty(M * Nrho * vsize, dtype=cdtype)
        self.d_t2 = gpuarray.empty_like(self.d_t1)
        self.d_t3 = gpuarray.empty_like(self.d_t1)
        self.d_t4 = gpuarray.empty_like(self.d_t1)

        self.block = (128, 1, 1)
        self.grid = get_grid_for_block(self.block, vsize)
        self.gridNrhoNv = get_grid_for_block(self.block, Nrho * vsize)
        self.gridNrhoMNv = get_grid_for_block(self.block, Nrho * M * vsize)

        # define complex to complex plan
        rank = 3
        n = np.array([N, N, N], dtype=np.int32)

        # planD2Z = cufftPlan3d(N, N, N, CUFFT_D2Z)
        self.planT2T_MNrho = cufftPlanMany(
            rank, n.ctypes.data, None, 1, vsize, None, 1, vsize, CUFFT_T2T, M * Nrho
        )
        self.planT2T = cufftPlan3d(N, N, N, CUFFT_T2T)

        dfltargs = dict(
            dtype=self.cfg.dtypename,
            Nrho=Nrho,
            M=M,
            vsize=vsize,
            sw=self.vm.sw(),
            prefac=self._prefactor,
            qw=qw,
            sz=sz,
            gamma=self._gamma,
            L=L,
            qz=qz,
            Ne=self._Ne,
            block_size=self.block[0],
            cw=self.vm.cw(),
        )
        src = (
            DottedTemplateLookup("edgfs2D.std.kernels.scattering", dfltargs)
            .get_template("vhs-gll")
            .render()
        )

        # Compile the source code and retrieve the kernel
        module = compiler.SourceModule(src)

        self.d_aa = gpuarray.empty(Nrho * M * vsize, dtype=dtype)
        precompute_aa = get_kernel(module, "precompute_aa", "PPPP")
        precompute_aa.prepared_call(
            self.grid, self.block, d_lx.ptr, d_ly.ptr, d_lz.ptr, self.d_aa.ptr
        )

        self.d_bb1 = gpuarray.empty(Nrho * vsize, dtype=dtype)
        self.d_bb2 = gpuarray.empty(vsize, dtype=dtype)
        precompute_bb = get_kernel(module, "precompute_bb", "PPPPP")
        precompute_bb.prepared_call(
            self.grid,
            self.block,
            d_lx.ptr,
            d_ly.ptr,
            d_lz.ptr,
            self.d_bb1.ptr,
            self.d_bb2.ptr,
        )

        # transform scalar to complex
        self.r2zKern = get_kernel(module, "r2z_", "iiPP")

        # Prepare the cosSinMul kernel for execution
        self.cosSinMultKern = get_kernel(module, "cosSinMul", "PPPP")

        # Prepare the magSqrKern kernel for execution
        self.magSqrKern = get_kernel(module, "magSqr", "PPP")

        # Prepare the computeQG kernel for execution
        self.computeQGKern = get_kernel(module, "computeQG", "PPP")

        # Prepare the ax kernel for execution
        self.axKern = get_kernel(module, "ax", "PP")

        # Prepare the output append kernel for execution
        self.outAppendKern = get_kernel(module, "output_append_", "iiiPPPP")

        # required by the child class (may be deleted by the child)
        self.module = module

        # compute nu
        self.nuKern = get_kernel(module, "nu", "iiPP")

        # compute nu2 (velocity dependent collision frequency)
        self.nu2Kern = get_kernel(module, "nu2", "iiPP")

        # sum kernel
        self.sumCplxKern = get_kernel(module, "sumCplx_", "PPII")
        self.sumKern = get_kernel(module, "sum_", "PPII")
        grid = self.grid
        block = self.block
        seq_count0 = int(4)
        N = int(vsize)
        # grid_count0 = int((grid[0] + (-grid[0] % seq_count0)) // seq_count0)
        grid_count0 = int((grid[0] // seq_count0 + ceil(grid[0] % seq_count0)))
        d_st = gpuarray.empty(grid_count0, dtype=dtype)
        d_nutmp = gpuarray.empty(1, dtype=dtype)
        # seq_count1 = int((grid_count0 + (-grid_count0 % block[0])) // block[0])
        seq_count1 = int((grid_count0 // block[0] + ceil(grid_count0 % block[0])))

        def sum_(d_in, d_out, elem, mode):
            self.sumCplxKern.prepared_call(
                (grid_count0, 1), block, d_in.ptr, d_st.ptr, seq_count0, N
            )
            self.sumKern.prepared_call(
                (1, 1), block, d_st.ptr, d_nutmp.ptr, seq_count1, grid_count0
            )
            self.nuKern.prepared_call(
                (1, 1), (1, 1, 1), elem, mode, d_nutmp.ptr, d_out.ptr
            )

        self.nuFunc = sum_

    # def fs(self, d_arr_in, d_arr_out, elem, modein, modeout):
    # def fs(self, d_arr_in, d_arr_in2, d_arr_out, elem, modein, modeout):
    def solve(self, d_arr_in, d_arr_in2, d_arr_out, elem, modein, modeout, d_nu=None):
        d_f0 = d_arr_in.ptr
        d_Q = d_arr_out.ptr

        # construct d_fC from d_f0
        self.r2zKern.prepared_call(
            self.grid, self.block, elem, modein, d_f0, self.d_fC.ptr
        )

        # compute forward FFT of f | Ftf = fft(f)
        self.cufftExecT2T(self.planT2T, self.d_fC.ptr, self.d_FTf.ptr, CUFFT_FORWARD)
        # self.scaleKern.prepared_call(self.grid, self.block,
        #    self.d_FTf.ptr)

        # compute t1_{pqr} = cos(a_{pqr})*FTf_r; t2_{pqr} = sin(a_{pqr})*FTf_r
        # scales d_FTf
        self.cosSinMultKern.prepared_call(
            self.grid,
            self.block,
            self.d_aa.ptr,
            self.d_FTf.ptr,
            self.d_t1.ptr,
            self.d_t2.ptr,
        )

        # compute inverse fft
        self.cufftExecT2T(
            self.planT2T_MNrho, self.d_t1.ptr, self.d_t3.ptr, CUFFT_INVERSE
        )
        self.cufftExecT2T(
            self.planT2T_MNrho, self.d_t2.ptr, self.d_t4.ptr, CUFFT_INVERSE
        )

        # compute t2 = t3^2 + t4^2
        self.magSqrKern.prepared_call(
            self.grid, self.block, self.d_t3.ptr, self.d_t4.ptr, self.d_t2.ptr
        )

        # compute t1 = fft(t2)
        self.cufftExecT2T(
            self.planT2T_MNrho, self.d_t2.ptr, self.d_t1.ptr, CUFFT_FORWARD
        )
        # scaling factor is multiplied in the computeQGKern
        # note: t1 is not modified in computeQGKern
        # self.scaleMNKern.prepared_call(self.grid, self.block,
        #    self.d_t1.ptr)

        # compute fC_r = 2*wrho_p*ws*b1_p*t1_r
        self.computeQGKern.prepared_call(
            self.grid, self.block, self.d_bb1.ptr, self.d_t1.ptr, self.d_fC.ptr
        )

        # inverse fft| QG = iff(fC)  [Gain computed]
        self.cufftExecT2T(self.planT2T, self.d_fC.ptr, self.d_QG.ptr, CUFFT_INVERSE)

        # compute FTf_r = b2_r*FTf_r
        self.axKern.prepared_call(self.grid, self.block, self.d_bb2.ptr, self.d_FTf.ptr)

        # inverse fft| fC = iff(FTf)
        self.cufftExecT2T(self.planT2T, self.d_FTf.ptr, self.d_fC.ptr, CUFFT_INVERSE)

        if d_nu:
            self.nuFunc(self.d_fC, d_nu, elem, modeout)
            # self.nu2Kern.prepared_call(self.grid, self.block,
            #    elem, modeout, self.d_fC.ptr, d_nu.ptr)

        # outKern
        self.outAppendKern.prepared_call(
            self.grid,
            self.block,
            elem,
            modein,
            modeout,
            self.d_QG.ptr,
            self.d_fC.ptr,
            d_f0,
            d_Q,
        )
