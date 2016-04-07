#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
using namespace thrust; 
#include <float.h>
#include <array>
#include <gl/freeglut.h>
#define NOMINMAX
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <lodepng.h>
#include <string> 
#include <vector> 
#include <string>
#include <string>
#include <fstream>
#include <sstream>
#include <streambuf>
#include <iostream>
#include <string>
#include <iterator>
#include <tchar.h>
#pragma comment(lib,"cudart")
#pragma comment(lib,"lodepng")
using namespace std;

#ifndef __CUDACC__
#error This file can only be compiled as a .cu file by nvcc.
#endif

#ifndef __CUDA_ARCH__
#define __CUDACC_RTC__ // HACK calm intellisense about __syncthreads et. al
#endif

#ifndef _WIN64
#error cudaMallocManaged and __managed__ require 64 bits. Also, this program is made for windows.
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 500
#error Always use the latest cuda arch. Old versions dont support any amount of thread blocks being submitted at once.
#endif

#ifdef __CUDA_ARCH__
#define GPU_CODE 1
#else
#define GPU_CODE 0
#endif

#define GPU_ONLY __device__
#define GPU(mem) mem
#define KERNEL __global__ void
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define CPU_AND_GPU __device__
#else
#define CPU_AND_GPU 
#endif























































#ifndef __CUDACC__
// HACK to make intellisense shut up about illegal C++ <<< >>>
#define LAUNCH_KERNEL(kernelFunction, gridDim, blockDim, arguments, ...) ((void)0)
#else
dim3 _lastLaunch_gridDim, _lastLaunch_blockDim;
#define LAUNCH_KERNEL(kernelFunction, gridDim, blockDim, ...) {\
cudaSafeCall(cudaGetLastError());\
_lastLaunch_gridDim = dim3(gridDim); _lastLaunch_blockDim = dim3(blockDim);\
kernelFunction << <gridDim, blockDim >> >(__VA_ARGS__);\
cudaSafeCall(cudaGetLastError());\
cudaSafeCall(cudaDeviceSynchronize()); /* TODO greatly alters the execution order */\
}

#endif






















































#undef assert
#if GPU_CODE
#define assert(x,commentFormat,...) if(!(x)) {printf("%s(%i) : Assertion failed : %s.\n\tblockIdx %d %d %d, threadIdx %d %d %d\n\t<" commentFormat ">\n", __FILE__, __LINE__, #x, xyz(blockIdx), xyz(threadIdx), __VA_ARGS__); *(int*)0 = 0;/* asm("trap;"); illegal instruction*/} 
#else
#define assert(x,commentFormat,...) if(!(x)) {char s[10000]; sprintf_s(s, "%s(%i) : Assertion failed : %s.\n\t<" commentFormat ">\n", __FILE__, __LINE__, #x, __VA_ARGS__); puts(s); flushStd(); DebugBreak(); OutputDebugStringA("! program continues after failed assertion\n\n");} 
#endif

























































// Automatically wrap some functions in cudaSafeCall
#ifdef __CUDACC__ // hack to hide these from intellisense
#define cudaDeviceSynchronize(...) cudaSafeCall(cudaDeviceSynchronize(__VA_ARGS__))
#define cudaMalloc(...) cudaSafeCall(cudaMalloc(__VA_ARGS__))
#define cudaFree(...) cudaSafeCall(cudaFree(__VA_ARGS__))
#define cudaMallocManaged(...) cudaSafeCall(cudaMallocManaged(__VA_ARGS__))
#endif




















































// cudaSafeCall is an expression that evaluates to 
// 0 when err is cudaSuccess (0), such that cudaSafeCall(cudaSafeCall(cudaSuccess)) will not block
// this is important because we might have legacy code that explicitly does 
// cudaSafeCall(cudaDeviceSynchronize());
// but we extended cudaDeviceSynchronize to include this already, giving
// cudaSafeCall(cudaSafeCall(cudaDeviceSynchronize()))

// it debug-breaks and returns 
bool cudaSafeCallImpl(cudaError err, const char * const expr, const char * const file, const int line);

// If err is cudaSuccess, cudaSafeCallImpl will return true, early-out of || will make DebugBreak not evaluated.
// The final expression will be 0.
// Otherwise we evaluate debug break, which returns true as well and then return 0.
#define cudaSafeCall(err) \
    !(cudaSafeCallImpl((cudaError)(err), #err, __FILE__, __LINE__) || ([]() {DebugBreak(); return true;})() )



















































#define xyz(p) p.x, p.y, p.z
#define xy(p) p.x, p.y
#define threadIdx_xyz xyz(threadIdx)























































#define stdouterrfile "stdouterr.txt"
void redirectStd() {
    freopen(stdouterrfile, "w", stdout); // stdout > stdoutfile, 1>stdoutfile
    freopen(stdouterrfile, "w", stderr); // 2>stderrfile
}






















































struct _atinit {
    _atinit() {
        redirectStd();
        // CATCHMALLOCERRORS
        _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF | _CRTDBG_CHECK_ALWAYS_DF);
    }
} __atinit;






















































struct _atexit {
    ~_atexit() {
        /// Catch remaining cuda errors on shutdown
        cudaSafeCall(cudaGetLastError());
        cudaSafeCall(cudaDeviceSynchronize());
        cudaSafeCall(cudaGetLastError());

        // TODO is this still true?
        // cudaDeviceReset must be called before exiting in order for profiling and
        // tracing tools such as Nsight and Visual Profiler to show complete traces.
        cudaDeviceReset();
    }
} __atexit;






















































int fileExists(TCHAR * file)
{
    WIN32_FIND_DATA FindFileData;
    HANDLE handle = FindFirstFile(file, &FindFileData);
    int found = handle != INVALID_HANDLE_VALUE;
    if (found)
    {
        //FindClose(&handle); this will crash
        FindClose(handle);
    }
    return found;
}






















































std::string readFile(std::string fn) {
    std::ifstream t(fn);
    return std::string(std::istreambuf_iterator<char>(t),
        std::istreambuf_iterator<char>());
}






















































void flushStd() {
    if (!fileExists(stdouterrfile)) return;
    // unlock stdout.txt & stderr
    ::fflush(stdout);
    ::fflush(stderr);
    freopen("CONOUT$", "w", stdout);
    freopen("CONOUT$", "w", stderr);
    std::string s = (
        "<<< " stdouterrfile " >>>\n" + readFile(stdouterrfile)
        );

    OutputDebugStringA(s.c_str());

    remove(stdouterrfile);
    redirectStd();
}






















































/// \returns true if err is cudaSuccess
/// Fills errmsg in UNIT_TESTING build.
bool cudaSafeCallImpl(cudaError err, const char * const expr, const char * const file, const int line)
{
    if (cudaSuccess == err) return true;

    char s[10000];
    cudaGetLastError(); // Reset error flag
    const char* e = cudaGetErrorString(err);
    if (!e) e = "! cudaGetErrorString returned 0 !";

    sprintf_s(s, "\n%s(%i) : cudaSafeCall(%s)\nRuntime API error : %s.\n",
        file,
        line,
        expr,
        e);
    puts(s);
    if (err == cudaError::cudaErrorLaunchFailure) {
        printf("maybe illegal memory access, (memcpy(0,0,4) et.al) try the CUDA debugger\n"
            );
    }

    if (err == cudaError::cudaErrorInvalidConfiguration) {
        printf("configuration was (%d,%d,%d), (%d,%d,%d)\n",
            xyz(_lastLaunch_gridDim),
            xyz(_lastLaunch_blockDim)
            );
    }

    if (err == cudaError::cudaErrorIllegalInstruction) {
        puts("maybe the illegal instruction was asm(trap;) of a failed assertion?");
    }


    flushStd();
    return false;
}






















































/*
* A cut-down local version of gluErrorString to avoid depending on GLU.
*/
const char* fghErrorString(GLenum error)
{
    switch (error) {
    case GL_INVALID_ENUM: return "invalid enumerant";
    case GL_INVALID_VALUE: return "invalid value";
    case GL_INVALID_OPERATION: return "invalid operation";
#ifndef GL_ES_VERSION_2_0
    case GL_STACK_OVERFLOW: return "stack overflow";
    case GL_STACK_UNDERFLOW: return "stack underflow";
#endif
    case GL_OUT_OF_MEMORY: return "out of memory";
    default: return "unknown GL error";
    }
}
























































__managed__ float* sum_Atb; // m x 1
__managed__ float* sum_AtA; // m x m, row major
__managed__ int n;






















































template<typename T, int b>
KERNEL k() {
    assert(false, "problem %d", 5);
}




















































#define REDUCE_BLOCK_SIZE 256
template<class Constructor,int m>
KERNEL constructAndSolve_device(int n) {
    assert(gridDim.y == gridDim.z && gridDim.y == 1);
    assert(blockDim.x == REDUCE_BLOCK_SIZE);
    assert(threadIdx.y == threadIdx.z && threadIdx.y == 0);

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    assert(n == ::n);

    __shared__ float ssum_ai_aiT[REDUCE_BLOCK_SIZE][m][m];// = {0};
    __shared__ float ssum_ai_bi[REDUCE_BLOCK_SIZE][m];// = {0};

    const int tid = threadIdx.x;
    if (i >= n) {

        memset(&ssum_ai_aiT[tid], 0, sizeof(float) * m *m);
        memset(&ssum_ai_bi[tid], 0, sizeof(float) * m);
        return;
    }

    // Computation
    float ai[m]; /* m x 1 */
    float bi;
    Constructor::generate<m>(i, ai, bi);

    // Summands
    float ai_aiT[m][m];
    for (int c = 0; c < m; c++)
        for (int r = 0; r < m; r++) 
            ai_aiT[c][r] = ai[c] * ai[r];

    float ai_bi[m]; /* m x 1 */
    for (int r = 0; r < m; r++)
        ai_bi[r] = ai[r] * bi;

    // -- Summation ---

    // FAST

    // TODO could write to these right away
    memcpy(&ssum_ai_aiT[tid], ai_aiT, sizeof(float) * m *m);
    memcpy(&ssum_ai_bi[tid], ai_bi, sizeof(float) * m);
    __syncthreads();

    // SLOW summation into sum_ai_aiT sum_ai_bi
    __shared__ float sum_ai_aiT[m][m];
    __shared__ float sum_ai_bi[m];
    
    if (tid == 0) {
        memset(&sum_ai_aiT, 0, sizeof(float) * m *m);
        memset(&sum_ai_bi, 0, sizeof(float) * m);

        for (int j = 0; j < REDUCE_BLOCK_SIZE; j++) {
            for (int c = 0; c < m; c++)
                for (int r = 0; r < m; r++)
                    sum_ai_aiT[c][r] += ssum_ai_aiT[j][c][r];

            for (int r = 0; r < m; r++)
                sum_ai_bi[r] += ssum_ai_bi[j][r];
        }
    }
    __syncthreads();

    // FAST summation into sum_ai_aiT[0] sum_ai_bi[0]
    for (int offset = REDUCE_BLOCK_SIZE / 2; offset >= 1; offset /= 2) {
        if (tid >= offset) return;

        for (int c = 0; c < m; c++)
            for (int r = 0; r < m; r++)
                ssum_ai_aiT[tid][c][r] += ssum_ai_aiT[tid + offset][c][r];

        for (int r = 0; r < m; r++)
            ssum_ai_bi[tid][r] += ssum_ai_bi[tid + offset][r];

        __syncthreads();
    }

    // check
    assert(tid == 0);

    if (blockIdx.x == 1) {
        printf("\n");
        for (int r = 0; r < m; r++) {
            for (int c = 0; c < m; c++) {
                printf("%.0f ", sum_ai_aiT[c][r]);
            }
            printf("\n");
        }

        printf("-----\n");
        for (int r = 0; r < m; r++) {
            for (int c = 0; c < m; c++) {
                printf("%.0f ", ssum_ai_aiT[0][c][r]);
            }
            printf("\n");
        }
    }

    for (int c = 0; c < m; c++)
        for (int r = 0; r < m; r++)
            assert(ssum_ai_aiT[0][c][r] == sum_ai_aiT[c][r], "%f %f", ssum_ai_aiT[0][c][r], sum_ai_aiT[c][r]);

    for (int r = 0; r < m; r++)
        assert(ssum_ai_bi[0][r] == sum_ai_bi[r], "%f %f", ssum_ai_bi[0][r], sum_ai_bi[r]);

    // Sum globally
    for (int c = 0; c < m; c++)
        for (int r = 0; r < m; r++)
            atomicAdd(&sum_AtA[r*m + c], ssum_ai_aiT[0][c][r]);

    for (int r = 0; r < m; r++)
        atomicAdd(&sum_Atb[r], ssum_ai_bi[0][r]);
}


















































float assertFinite(float value) {
    assert(_fpclass(value) == _FPCLASS_PD || _fpclass(value) == _FPCLASS_PN || _fpclass(value) == _FPCLASS_PZ ||
        _fpclass(value) == _FPCLASS_ND || _fpclass(value) == _FPCLASS_NN || _fpclass(value) == _FPCLASS_NZ
        , "value = %f is not finite", value);
    return value;
}
















































class Cholesky
{
private:
    std::vector<float> cholesky;
    int size, rank;

public:
    // Solve Ax = b for A symmetric positive-definite of size*size
    static void solve(const float* mat, int size, const float* b, float* result) {
        Cholesky cholA(mat, size);
        cholA.Backsub(result, b);
    }

    /// \f[A = LL*\f]
    /// Produces Cholesky decomposition of the
    /// symmetric, positive-definite matrix mat of dimension size*size
    /// \f$L\f$ is a lower triangular matrix with real and positive diagonal entries
    /// assertFinite is used to detect singular matrices and other non-supported cases.
    Cholesky(const float *mat, int size)
    {
        this->size = size;
        this->cholesky.resize(size*size);

        for (int i = 0; i < size * size; i++) cholesky[i] = assertFinite(mat[i]);

        for (int c = 0; c < size; c++)
        {
            float inv_diag = 1;
            for (int r = c; r < size; r++)
            {
                float val = cholesky[c + r * size];
                for (int c2 = 0; c2 < c; c2++)
                    val -= cholesky[c + c2 * size] * cholesky[c2 + r * size];

                if (r == c)
                {
                    cholesky[c + r * size] = assertFinite(val);
                    if (val == 0) { rank = r; }
                    inv_diag = 1.0f / val;
                }
                else
                {
                    cholesky[r + c * size] = assertFinite(val);
                    cholesky[c + r * size] = assertFinite(val * inv_diag);
                }
            }
        }

        rank = size;
    }

    /// Solves \f[Ax = b\f]
    /// by
    /// * solving Ly = b for y by forward substitution, and then
    /// * solving L*x = y for x by back substitution.
    void Backsub(
        float *x,  //!< out \f$x\f$
        const float *b //!< input \f$b\f$
        ) const
    {
        // Forward
        std::vector<float> y(size);
        for (int i = 0; i < size; i++)
        {
            float val = b[i];
            for (int j = 0; j < i; j++) val -= cholesky[j + i * size] * y[j];
            y[i] = val;
        }

        for (int i = 0; i < size; i++) y[i] /= cholesky[i + i * size];

        // Backward
        for (int i = size - 1; i >= 0; i--)
        {
            float val = y[i];
            for (int j = i + 1; j < size; j++) val -= cholesky[i + j * size] * x[j];
            x[i] = val;
        }
    }
};






















































/** Allocate a block of CUDA memory and memset it to 0 */
template<typename T> static void zeroManagedMalloc(T*& p, const unsigned int count = 1) {
    cudaSafeCall(cudaMallocManaged(&p, sizeof(T) * count));
    cudaSafeCall(cudaMemset(p, 0, sizeof(T) * count));
}



















































template<class Constructor, int m>
std::array<float, m> constructAndSolve(int n) {
    assert(m < 100);
    ::n = n;

    cudaDeviceSynchronize();
    zeroManagedMalloc(sum_AtA, m * m);
    zeroManagedMalloc(sum_Atb, m);
    assert(sum_AtA[0] == 0);
    assert(sum_Atb[0] == 0);
    assert(sum_AtA[m * m-1] == 0);
    assert(sum_Atb[m - 1] == 0);

    LAUNCH_KERNEL(
        (constructAndSolve_device<Constructor, m>),
        ceil(n / (1.f * REDUCE_BLOCK_SIZE)),
        REDUCE_BLOCK_SIZE,
        n);
    cudaDeviceSynchronize();

    for (int r = 0; r < m; r++) {
        puts("");
        for (int c = 0; c < m; c++)
            cout << sum_AtA[r * m+ c] << " ";
    }
    assert(sum_AtA[m+1] != 0);
    assert(sum_Atb[m-1] != 0);

    auto x = std::array < float, m >();
    Cholesky::solve(sum_AtA, m, sum_Atb, x.data());
    return x;
}

























































struct ConstructExampleEquation {
    template<int m>
    static GPU_ONLY void generate(const int i, float out_ai[m], float& out_bi/*[1]*/) {
        for (int j = 0; j < m; j++) {
            out_ai[j] = 0;
            if (i == j || i == 0|| j== 0)
                out_ai[j] = 1;
        }
        out_bi = i+1;
    }
};











































void assertApproxEqual(float a, float b, int considered_initial_bits = 20) {
    assert(considered_initial_bits > 8 + 1); // should consider at least sign and full exponent
    assert(considered_initial_bits <= 32);

    unsigned int ai = *(unsigned int*)&a;
    unsigned int bi = *(unsigned int*)&b;
    auto ait = ai >> (32 - considered_initial_bits);
    auto bit = bi >> (32 - considered_initial_bits);

    assert(ait == bit, "%f != %f, %x != %x, %x != %x", 
        a,b,ai,bi,ait,bit
        );
}

struct Trafo {
    CPU_AND_GPU int operator()(int i) {
        return 2 * i;
    }
};
int main(int argc, char** argv)
{
    const int min = 1, max = 6;
    thrust::counting_iterator<int> first(min);
    auto last = first + (max-min) + 1;
    int res = transform_reduce(first, last, Trafo(), 0, thrust::plus<int>());
    assert(res == 2 * (max * (max + 1)) / 2);

    const int m = 6;
    const int n = 2*REDUCE_BLOCK_SIZE;
    auto x = constructAndSolve<ConstructExampleEquation, m>(n);
    float expect[m] = {258.164, -87.2215, -86.2215, -85.2215, -84.2215, -83.2215};
    for (int i = 0; i < m; i++)
        assertApproxEqual(x[i], expect[i]);

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(400, 300);
    glutCreateWindow("Hello World");
    auto e = glGetError();
    glGenTextures(-10,0);
    auto e2 = glGetError(); 
    
    ((void)0);//LAUNCH_KERNEL((k<int, 5>), 1, 1);

    return 0;
}