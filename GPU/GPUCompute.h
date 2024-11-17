/*
* Modified version work
*/

// CUDA Kernel main function

// Jump distance
__device__ __constant__ uint64_t jD[NB_JUMP][4];
// jump points
__device__ __constant__ uint64_t jPx[NB_JUMP][4];
__device__ __constant__ uint64_t jPy[NB_JUMP][4];


// Jump distance
__device__ __shared__ uint64_t  SjD[NB_JUMP][4];
// jump points
__device__ __shared__ uint64_t  SjPx[NB_JUMP][4];
__device__ __shared__ uint64_t  SjPy[NB_JUMP][4];

// -----------------------------------------------------------------------------------------


__device__ void ComputeKangaroos(uint64_t* kangaroos, uint32_t maxFound, uint32_t* out, uint64_t dpMask) {

	uint64_t px[GPU_GRP_SIZE][4];
	uint64_t py[GPU_GRP_SIZE][4];
	uint64_t dist[GPU_GRP_SIZE][4];
	uint64_t lastJump[GPU_GRP_SIZE];

	uint64_t dx[GPU_GRP_SIZE][4];
	uint64_t dy[4];
	uint64_t rx[4];
	uint64_t ry[4];
	uint64_t _s[4];
	uint64_t _p[4];
	uint64_t inverse[5];
	uint64_t jmp;

	int stride = threadIdx.x & 31;

	Load256(SjPx[stride], jPx[stride]);
	Load256(SjPy[stride], jPy[stride]);
	Load256(SjD[stride], jD[stride]);

	__syncthreads();
	LoadKangaroos(kangaroos, px, py, dist, lastJump);

	for (int run = 0; run < NB_RUN; run++) {
		//__syncthreads();
		inverse[3] = 0;
		inverse[2] = 0;
		inverse[1] = 0;
		inverse[0] = 1;


		for (int g = 0; g < GPU_GRP_SIZE; g++) {
			jmp = (unsigned int)px[g][0] & (NB_JUMP - 1);

			ModSub256(_p, px[g], SjPx[jmp]);

			_ModMult(inverse, inverse, _p);
			Load256(dx[g], inverse);
		}
		inverse[4] = 0;
		_ModInv(inverse);
		__syncthreads();


		for (int g = GPU_GRP_SIZE - 1; g >= 0; g--) {

			jmp = (unsigned int)px[g][0] & (NB_JUMP - 1);
			if (g >= 1) {
				_ModMult(_s, inverse, dx[g - 1]);

				ModSub256(_p, px[g], SjPx[jmp]);

				_ModMult(inverse, inverse, _p);

			}
			else {
				Load256(_s, inverse);
			}


			ModSub256(dy, py[g], SjPy[jmp]);
			_ModMult(_s, dy, _s);
			_ModSqr(_p, _s);

			ModSub256(rx, _p, SjPx[jmp]);
			ModSub256(rx, px[g]);

			ModSub256(ry, px[g], rx);
			_ModMult(ry, _s);
			ModSub256(ry, py[g]);


			__syncthreads();
			Load256(px[g], rx);
			Load256(py[g], ry);

			Add128(dist[g], SjD[jmp]);

			if ((px[g][3] & dpMask) == 0) {

				// Distinguished point
				uint32_t pos = atomicAdd(out, 1);
				if (pos < maxFound) {
					uint64_t kIdx = (uint64_t)IDX + (uint64_t)g * (uint64_t)blockDim.x + (uint64_t)blockIdx.x * ((uint64_t)blockDim.x * GPU_GRP_SIZE);
					OutputDP(px[g], dist[g], &kIdx);
				}

			}



		}

	}

	__syncthreads();
	StoreKangaroos(kangaroos, px, py, dist, lastJump);

}
