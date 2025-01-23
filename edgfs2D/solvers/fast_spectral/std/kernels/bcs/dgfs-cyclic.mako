#define scalar ${dtype}

__global__ void applyBC
(
    const scalar* __restrict__ ul,
    scalar* __restrict__ ur,
    const scalar* __restrict__ cvx
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<${vsize}) 
    {
        //scalar fac = (cvx[idx])*(${nl});    
        ur[idx] = ul[idx];
        //ur[idx] = (fac>=0)*(ur[idx]) + (fac<0)*(ul[idx]);
        //ur[idx] = (fac>=0)*(ul[idx]) + (fac<0)*(ur[idx]);
        //ur[idx] = (fac<0)*(0.) + (fac>=0)*(ul[idx]);
        //ur[idx] = (fac<0)*(0.) + (fac>=0)*(ul[idx]);
    }
}