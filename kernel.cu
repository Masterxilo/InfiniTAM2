#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/tuple.h>
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
#include <tuple>
#include <string>
#include <fstream>
#include <sstream>
#include <streambuf>
#include <iostream>
#include <string>
#include <exception>
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




















#define TEST(name) void name(); struct T##name {T##name() {name();}} _T##name; void name() 


































dim3 _lastLaunch_gridDim, _lastLaunch_blockDim;
#ifndef __CUDACC__
// HACK to make intellisense shut up about illegal C++ <<< >>>
#define LAUNCH_KERNEL(kernelFunction, gridDim, blockDim, arguments, ...) ((void)0)
#else
#define LAUNCH_KERNEL(kernelFunction, gridDim, blockDim, ...) {\
cudaSafeCall(cudaGetLastError());\
_lastLaunch_gridDim = dim3(gridDim); _lastLaunch_blockDim = dim3(blockDim);\
kernelFunction << <gridDim, blockDim >> >(__VA_ARGS__);\
cudaSafeCall(cudaGetLastError());\
cudaSafeCall(cudaDeviceSynchronize()); /* TODO greatly alters the execution order */\
}

#endif



















































/// Whether a failed assertion triggers a debugger break.
__managed__ bool breakOnAssertion = true;
/// Whenever an assertions fails std::exception is thrown.
/// No effect in GPU code.
bool assertionThrowException = false;
/// Must be reset manually
__managed__ bool assertionFailed = false;

#undef assert
#if GPU_CODE
#define assert(x,commentFormat,...) if(!(x)) {printf("%s(%i) : Assertion failed : %s.\n\tblockIdx %d %d %d, threadIdx %d %d %d\n\t<" commentFormat ">\n", __FILE__, __LINE__, #x, xyz(blockIdx), xyz(threadIdx), __VA_ARGS__); assertionFailed = true; if (breakOnAssertion) *(int*)0 = 0;/* asm("trap;"); illegal instruction*/} 
#else
#define assert(x,commentFormat,...) if(!(x)) {char s[10000]; sprintf_s(s, "%s(%i) : Assertion failed : %s.\n\t<" commentFormat ">\n", __FILE__, __LINE__, #x, __VA_ARGS__); puts(s); flushStd(); assertionFailed = true; if (breakOnAssertion) DebugBreak(); OutputDebugStringA("! program continues after failed assertion\n\n"); if (assertionThrowException) throw std::exception();} 
#endif





#define BEGIN_SHOULD_FAIL() {bool ok = false; assertionThrowException = true; breakOnAssertion = false; try {
#define END_SHOULD_FAIL() } catch(...) {assertionThrowException = false; breakOnAssertion = true; ok = true;} assert(ok);}






















































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
    !(cudaSafeCallImpl((cudaError)(err), #err, __FILE__, __LINE__) || ([]() {assert(false, "CUDA error"); return true;})() )



















































#define xyz(p) p.x, p.y, p.z
#define xy(p) p.x, p.y
#define threadIdx_xyz xyz(threadIdx)











































struct integer {
private:
    int value;
    int assertRange(long long val) {
        assert(val <= (long long)MAX_INT, "%lld is too big for an int");
        assert(val >= (long long)MIN_INT, "%lld is too small for an int");
        return val;
    }
public:
    integer(const int& value) : value(value) {}
    
    integer(const long long& value) : value(value) {}
    
    // deny:
    integer(float);
    integer(double);
    integer(unsigned int);
    integer(unsigned long);
    integer(unsigned long long);
    integer(long);
    
    const integer& operator=(const integer& b) {
        this->value = b.value;
    }
    
    #define operation(OP) \
    const integer& operator OP =(integer& b) {\
        *this = *this OP b;\
        return *this;\
    }\
    \
    integer operator OP (const integer& b_) {\
        long long a = this->value;\
        long long b = b_.value;\
        long long result = a OP b;\
        return integer(result);\
    }
     
};


struct uinteger {

};


TEST(underflow) {
    BEGIN_SHOULD_FAIL()
        uinteger x = 0;
        x--;
    END_SHOULD_FAIL()
}
































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
template<class Constructor, int m>
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







































































































/** Allocate a block of CUDA memory and memset it to 0 */
template<typename T> static void zeroManagedMalloc(T*& p, const unsigned int count = 1) {
    cudaSafeCall(cudaMallocManaged(&p, sizeof(T) * count));
    cudaSafeCall(cudaMemset(p, 0, sizeof(T) * count));
}





























//////////////////////////////////////////////////////////////////////////
//						Basic Vector Structure
//////////////////////////////////////////////////////////////////////////

template <class T> struct Vector2_{
    union {
        struct { T x, y; }; // standard names for components
        struct { T s, t; }; // standard names for components
        struct { T width, height; };
        T v[2];     // array access
    };
};

template <class T> struct Vector3_{
    union {
        struct{ T x, y, z; }; // standard names for components
        struct{ T r, g, b; }; // standard names for components
        struct{ T s, t, p; }; // standard names for components
        T v[3];
    };
};

template <class T> struct Vector4_ {
    union {
        struct { T x, y, z, w; }; // standard names for components
        struct { T r, g, b, a; }; // standard names for components
        struct { T s, t, p, q; }; // standard names for components
        T v[4];
    };
};

template <class T> struct Vector6_ {
    //union {
    T v[6];
    //};
};

template<class T, int s> struct VectorX_
{
    int vsize;
    T v[s];
};

//////////////////////////////////////////////////////////////////////////
// Vector class with math operators: +, -, *, /, +=, -=, /=, [], ==, !=, T*(), etc.
//////////////////////////////////////////////////////////////////////////
template <class T> class Vector2 : public Vector2_ < T >
{
public:
    typedef T value_type;
    CPU_AND_GPU inline int size() const { return 2; }

    ////////////////////////////////////////////////////////
    //  Constructors
    ////////////////////////////////////////////////////////
    CPU_AND_GPU Vector2(){} // Default constructor
    CPU_AND_GPU Vector2(const T &t) { this->x = t; this->y = t; } // Scalar constructor
    CPU_AND_GPU Vector2(const T *tp) { this->x = tp[0]; this->y = tp[1]; } // Construct from array			            
    CPU_AND_GPU Vector2(const T v0, const T v1) { this->x = v0; this->y = v1; } // Construct from explicit values
    CPU_AND_GPU Vector2(const Vector2_<T> &v) { this->x = v.x; this->y = v.y; }// copy constructor

    CPU_AND_GPU explicit Vector2(const Vector3_<T> &u)  { this->x = u.x; this->y = u.y; }
    CPU_AND_GPU explicit Vector2(const Vector4_<T> &u)  { this->x = u.x; this->y = u.y; }

    CPU_AND_GPU inline Vector2<int> toInt() const {
        return Vector2<int>((int)ROUND(this->x), (int)ROUND(this->y));
    }

    CPU_AND_GPU inline Vector2<int> toIntFloor() const {
        return Vector2<int>((int)floor(this->x), (int)floor(this->y));
    }

    CPU_AND_GPU inline Vector2<unsigned char> toUChar() const {
        Vector2<int> vi = toInt(); return Vector2<unsigned char>((unsigned char)CLAMP(vi.x, 0, 255), (unsigned char)CLAMP(vi.y, 0, 255));
    }

    CPU_AND_GPU inline Vector2<float> toFloat() const {
        return Vector2<float>((float)this->x, (float)this->y);
    }

    CPU_AND_GPU const T *getValues() const { return this->v; }
    CPU_AND_GPU Vector2<T> &setValues(const T *rhs) { this->x = rhs[0]; this->y = rhs[1]; return *this; }

    CPU_AND_GPU T area() const {
        return width * height;
    }
    // indexing operators
    CPU_AND_GPU T &operator [](int i) { return this->v[i]; }
    CPU_AND_GPU const T &operator [](int i) const { return this->v[i]; }

    // type-cast operators
    CPU_AND_GPU operator T *() { return this->v; }
    CPU_AND_GPU operator const T *() const { return this->v; }

    ////////////////////////////////////////////////////////
    //  Math operators
    ////////////////////////////////////////////////////////

    // scalar multiply assign
    CPU_AND_GPU friend Vector2<T> &operator *= (const Vector2<T> &lhs, T d) {
        lhs.x *= d; lhs.y *= d; return lhs;
    }

    // component-wise vector multiply assign
    CPU_AND_GPU friend Vector2<T> &operator *= (Vector2<T> &lhs, const Vector2<T> &rhs) {
        lhs.x *= rhs.x; lhs.y *= rhs.y; return lhs;
    }

    // scalar divide assign
    CPU_AND_GPU friend Vector2<T> &operator /= (Vector2<T> &lhs, T d) {
        if (d == 0) return lhs; lhs.x /= d; lhs.y /= d; return lhs;
    }

    // component-wise vector divide assign
    CPU_AND_GPU friend Vector2<T> &operator /= (Vector2<T> &lhs, const Vector2<T> &rhs) {
        lhs.x /= rhs.x; lhs.y /= rhs.y;	return lhs;
    }

    // component-wise vector add assign
    CPU_AND_GPU friend Vector2<T> &operator += (Vector2<T> &lhs, const Vector2<T> &rhs) {
        lhs.x += rhs.x; lhs.y += rhs.y;	return lhs;
    }

    // component-wise vector subtract assign
    CPU_AND_GPU friend Vector2<T> &operator -= (Vector2<T> &lhs, const Vector2<T> &rhs) {
        lhs.x -= rhs.x; lhs.y -= rhs.y;	return lhs;
    }

    // unary negate
    CPU_AND_GPU friend Vector2<T> operator - (const Vector2<T> &rhs) {
        Vector2<T> rv;	rv.x = -rhs.x; rv.y = -rhs.y; return rv;
    }

    // vector add
    CPU_AND_GPU friend Vector2<T> operator + (const Vector2<T> &lhs, const Vector2<T> &rhs)  {
        Vector2<T> rv(lhs); return rv += rhs;
    }

    // vector subtract
    CPU_AND_GPU friend Vector2<T> operator - (const Vector2<T> &lhs, const Vector2<T> &rhs) {
        Vector2<T> rv(lhs); return rv -= rhs;
    }

    // scalar multiply
    CPU_AND_GPU friend Vector2<T> operator * (const Vector2<T> &lhs, T rhs) {
        Vector2<T> rv(lhs); return rv *= rhs;
    }

    // scalar multiply
    CPU_AND_GPU friend Vector2<T> operator * (T lhs, const Vector2<T> &rhs) {
        Vector2<T> rv(lhs); return rv *= rhs;
    }

    // vector component-wise multiply
    CPU_AND_GPU friend Vector2<T> operator * (const Vector2<T> &lhs, const Vector2<T> &rhs) {
        Vector2<T> rv(lhs); return rv *= rhs;
    }

    // scalar multiply
    CPU_AND_GPU friend Vector2<T> operator / (const Vector2<T> &lhs, T rhs) {
        Vector2<T> rv(lhs); return rv /= rhs;
    }

    // vector component-wise multiply
    CPU_AND_GPU friend Vector2<T> operator / (const Vector2<T> &lhs, const Vector2<T> &rhs) {
        Vector2<T> rv(lhs); return rv /= rhs;
    }

    ////////////////////////////////////////////////////////
    //  Comparison operators
    ////////////////////////////////////////////////////////

    // equality
    CPU_AND_GPU friend bool operator == (const Vector2<T> &lhs, const Vector2<T> &rhs) {
        return (lhs.x == rhs.x) && (lhs.y == rhs.y);
    }

    // inequality
    CPU_AND_GPU friend bool operator != (const Vector2<T> &lhs, const Vector2<T> &rhs) {
        return (lhs.x != rhs.x) || (lhs.y != rhs.y);
    }


    friend std::ostream& operator<<(std::ostream& os, const Vector2<T>& dt){
        os << dt.x << ", " << dt.y;
        return os;
    }
};

template <class T> class Vector3 : public Vector3_ < T >
{
public:
    typedef T value_type;
    CPU_AND_GPU inline int size() const { return 3; }

    ////////////////////////////////////////////////////////
    //  Constructors
    ////////////////////////////////////////////////////////
    CPU_AND_GPU Vector3(){} // Default constructor
    CPU_AND_GPU Vector3(const T &t)	{ this->x = t; this->y = t; this->z = t; } // Scalar constructor
    CPU_AND_GPU Vector3(const T *tp) { this->x = tp[0]; this->y = tp[1]; this->z = tp[2]; } // Construct from array
    CPU_AND_GPU Vector3(const T v0, const T v1, const T v2) { this->x = v0; this->y = v1; this->z = v2; } // Construct from explicit values
    CPU_AND_GPU explicit Vector3(const Vector4_<T> &u)	{ this->x = u.x; this->y = u.y; this->z = u.z; }
    CPU_AND_GPU explicit Vector3(const Vector2_<T> &u, T v0) { this->x = u.x; this->y = u.y; this->z = v0; }

    CPU_AND_GPU inline Vector3<int> toIntRound() const {
        return Vector3<int>((int)ROUND(this->x), (int)ROUND(this->y), (int)ROUND(this->z));
    }

    CPU_AND_GPU inline Vector3<int> toInt() const {
        return Vector3<int>((int)(this->x), (int)(this->y), (int)(this->z));
    }

    CPU_AND_GPU inline Vector3<int> toInt(Vector3<float> &residual) const {
        Vector3<int> intRound = toInt();
        residual = Vector3<float>(this->x - intRound.x, this->y - intRound.y, this->z - intRound.z);
        return intRound;
    }

    CPU_AND_GPU inline Vector3<short> toShortRound() const {
        return Vector3<short>((short)ROUND(this->x), (short)ROUND(this->y), (short)ROUND(this->z));
    }

    CPU_AND_GPU inline Vector3<short> toShortFloor() const {
        return Vector3<short>((short)floor(this->x), (short)floor(this->y), (short)floor(this->z));
    }

    CPU_AND_GPU inline Vector3<int> toIntFloor() const {
        return Vector3<int>((int)floor(this->x), (int)floor(this->y), (int)floor(this->z));
    }

    /// Floors the coordinates to integer values, returns this and the residual float.
    /// Use like
    /// TO_INT_FLOOR3(int_xyz, residual_xyz, xyz)
    /// for xyz === this
    CPU_AND_GPU inline Vector3<int> toIntFloor(Vector3<float> &residual) const {
        Vector3<float> intFloor(floor(this->x), floor(this->y), floor(this->z));
        residual = *this - intFloor;
        return Vector3<int>((int)intFloor.x, (int)intFloor.y, (int)intFloor.z);
    }

    CPU_AND_GPU inline Vector3<unsigned char> toUChar() const {
        Vector3<int> vi = toIntRound(); return Vector3<unsigned char>((unsigned char)CLAMP(vi.x, 0, 255), (unsigned char)CLAMP(vi.y, 0, 255), (unsigned char)CLAMP(vi.z, 0, 255));
    }

    CPU_AND_GPU inline Vector3<float> toFloat() const {
        return Vector3<float>((float)this->x, (float)this->y, (float)this->z);
    }

    CPU_AND_GPU inline Vector3<float> normalised() const {
        float norm = 1.0f / sqrt((float)(this->x * this->x + this->y * this->y + this->z * this->z));
        return Vector3<float>((float)this->x * norm, (float)this->y * norm, (float)this->z * norm);
    }

    CPU_AND_GPU const T *getValues() const	{ return this->v; }
    CPU_AND_GPU Vector3<T> &setValues(const T *rhs) { this->x = rhs[0]; this->y = rhs[1]; this->z = rhs[2]; return *this; }

    // indexing operators
    CPU_AND_GPU T &operator [](int i) { return this->v[i]; }
    CPU_AND_GPU const T &operator [](int i) const { return this->v[i]; }

    // type-cast operators
    CPU_AND_GPU operator T *()	{ return this->v; }
    CPU_AND_GPU operator const T *() const { return this->v; }

    ////////////////////////////////////////////////////////
    //  Math operators
    ////////////////////////////////////////////////////////

    // scalar multiply assign
    CPU_AND_GPU friend Vector3<T> &operator *= (Vector3<T> &lhs, T d)	{
        lhs.x *= d; lhs.y *= d; lhs.z *= d; return lhs;
    }

    // component-wise vector multiply assign
    CPU_AND_GPU friend Vector3<T> &operator *= (Vector3<T> &lhs, const Vector3<T> &rhs) {
        lhs.x *= rhs.x; lhs.y *= rhs.y; lhs.z *= rhs.z; return lhs;
    }

    // scalar divide assign
    CPU_AND_GPU friend Vector3<T> &operator /= (Vector3<T> &lhs, T d) {
        lhs.x /= d; lhs.y /= d; lhs.z /= d; return lhs;
    }

    // component-wise vector divide assign
    CPU_AND_GPU friend Vector3<T> &operator /= (Vector3<T> &lhs, const Vector3<T> &rhs)	{
        lhs.x /= rhs.x; lhs.y /= rhs.y; lhs.z /= rhs.z; return lhs;
    }

    // component-wise vector add assign
    CPU_AND_GPU friend Vector3<T> &operator += (Vector3<T> &lhs, const Vector3<T> &rhs)	{
        lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z; return lhs;
    }

    // component-wise vector subtract assign
    CPU_AND_GPU friend Vector3<T> &operator -= (Vector3<T> &lhs, const Vector3<T> &rhs) {
        lhs.x -= rhs.x; lhs.y -= rhs.y; lhs.z -= rhs.z; return lhs;
    }

    // unary negate
    CPU_AND_GPU friend Vector3<T> operator - (const Vector3<T> &rhs)	{
        Vector3<T> rv; rv.x = -rhs.x; rv.y = -rhs.y; rv.z = -rhs.z; return rv;
    }

    // vector add
    CPU_AND_GPU friend Vector3<T> operator + (const Vector3<T> &lhs, const Vector3<T> &rhs){
        Vector3<T> rv(lhs); return rv += rhs;
    }

    // vector subtract
    CPU_AND_GPU friend Vector3<T> operator - (const Vector3<T> &lhs, const Vector3<T> &rhs){
        Vector3<T> rv(lhs); return rv -= rhs;
    }

    // scalar multiply
    CPU_AND_GPU friend Vector3<T> operator * (const Vector3<T> &lhs, T rhs) {
        Vector3<T> rv(lhs); return rv *= rhs;
    }

    // scalar multiply
    CPU_AND_GPU friend Vector3<T> operator * (T lhs, const Vector3<T> &rhs) {
        Vector3<T> rv(lhs); return rv *= rhs;
    }

    // vector component-wise multiply
    CPU_AND_GPU friend Vector3<T> operator * (const Vector3<T> &lhs, const Vector3<T> &rhs)	{
        Vector3<T> rv(lhs); return rv *= rhs;
    }

    // scalar multiply
    CPU_AND_GPU friend Vector3<T> operator / (const Vector3<T> &lhs, T rhs) {
        Vector3<T> rv(lhs); return rv /= rhs;
    }

    // vector component-wise multiply
    CPU_AND_GPU friend Vector3<T> operator / (const Vector3<T> &lhs, const Vector3<T> &rhs) {
        Vector3<T> rv(lhs); return rv /= rhs;
    }

    ////////////////////////////////////////////////////////
    //  Comparison operators
    ////////////////////////////////////////////////////////

    // inequality
    CPU_AND_GPU friend bool operator != (const Vector3<T> &lhs, const Vector3<T> &rhs) {
        return (lhs.x != rhs.x) || (lhs.y != rhs.y) || (lhs.z != rhs.z);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // dimension specific operations
    ////////////////////////////////////////////////////////////////////////////////


    friend std::ostream& operator<<(std::ostream& os, const Vector3<T>& dt){
        os << dt.x << ", " << dt.y << ", " << dt.z;
        return os;
    }
};

////////////////////////////////////////////////////////
//  Non-member comparison operators
////////////////////////////////////////////////////////

// equality
template <typename T1, typename T2> CPU_AND_GPU inline bool operator == (const Vector3<T1> &lhs, const Vector3<T2> &rhs){
    return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z);
}

template <class T> class Vector4 : public Vector4_ < T >
{
public:
    typedef T value_type;
    CPU_AND_GPU inline int size() const { return 4; }

    ////////////////////////////////////////////////////////
    //  Constructors
    ////////////////////////////////////////////////////////

    CPU_AND_GPU Vector4() {} // Default constructor
    CPU_AND_GPU Vector4(const T &t) { this->x = t; this->y = t; this->z = t; this->w = t; } //Scalar constructor
    CPU_AND_GPU Vector4(const T *tp) { this->x = tp[0]; this->y = tp[1]; this->z = tp[2]; this->w = tp[3]; } // Construct from array
    CPU_AND_GPU Vector4(const T v0, const T v1, const T v2, const T v3) { this->x = v0; this->y = v1; this->z = v2; this->w = v3; } // Construct from explicit values
    CPU_AND_GPU explicit Vector4(const Vector3_<T> &u, T v0) { this->x = u.x; this->y = u.y; this->z = u.z; this->w = v0; }
    CPU_AND_GPU explicit Vector4(const Vector2_<T> &u, T v0, T v1) { this->x = u.x; this->y = u.y; this->z = v0; this->w = v1; }

    CPU_AND_GPU inline Vector4<int> toIntRound() const {
        return Vector4<int>((int)ROUND(this->x), (int)ROUND(this->y), (int)ROUND(this->z), (int)ROUND(this->w));
    }

    CPU_AND_GPU inline Vector4<unsigned char> toUChar() const {
        Vector4<int> vi = toIntRound(); return Vector4<unsigned char>((unsigned char)CLAMP(vi.x, 0, 255), (unsigned char)CLAMP(vi.y, 0, 255), (unsigned char)CLAMP(vi.z, 0, 255), (unsigned char)CLAMP(vi.w, 0, 255));
    }

    CPU_AND_GPU inline Vector4<float> toFloat() const {
        return Vector4<float>((float)this->x, (float)this->y, (float)this->z, (float)this->w);
    }

    CPU_AND_GPU inline Vector4<T> homogeneousCoordinatesNormalize() const {
        return (this->w <= 0) ? *this : Vector4<T>(this->x / this->w, this->y / this->w, this->z / this->w, 1);
    }

    CPU_AND_GPU inline Vector3<T> toVector3() const {
        return Vector3<T>(this->x, this->y, this->z);
    }

    CPU_AND_GPU const T *getValues() const { return this->v; }
    CPU_AND_GPU Vector4<T> &setValues(const T *rhs) { this->x = rhs[0]; this->y = rhs[1]; this->z = rhs[2]; this->w = rhs[3]; return *this; }

    // indexing operators
    CPU_AND_GPU T &operator [](int i) { return this->v[i]; }
    CPU_AND_GPU const T &operator [](int i) const { return this->v[i]; }

    // type-cast operators
    CPU_AND_GPU operator T *() { return this->v; }
    CPU_AND_GPU operator const T *() const { return this->v; }

    ////////////////////////////////////////////////////////
    //  Math operators
    ////////////////////////////////////////////////////////

    // scalar multiply assign
    CPU_AND_GPU friend Vector4<T> &operator *= (Vector4<T> &lhs, T d) {
        lhs.x *= d; lhs.y *= d; lhs.z *= d; lhs.w *= d; return lhs;
    }

    // component-wise vector multiply assign
    CPU_AND_GPU friend Vector4<T> &operator *= (Vector4<T> &lhs, const Vector4<T> &rhs) {
        lhs.x *= rhs.x; lhs.y *= rhs.y; lhs.z *= rhs.z; lhs.w *= rhs.w; return lhs;
    }

    // scalar divide assign
    CPU_AND_GPU friend Vector4<T> &operator /= (Vector4<T> &lhs, T d){
        lhs.x /= d; lhs.y /= d; lhs.z /= d; lhs.w /= d; return lhs;
    }

    // component-wise vector divide assign
    CPU_AND_GPU friend Vector4<T> &operator /= (Vector4<T> &lhs, const Vector4<T> &rhs) {
        lhs.x /= rhs.x; lhs.y /= rhs.y; lhs.z /= rhs.z; lhs.w /= rhs.w; return lhs;
    }

    // component-wise vector add assign
    CPU_AND_GPU friend Vector4<T> &operator += (Vector4<T> &lhs, const Vector4<T> &rhs)	{
        lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z; lhs.w += rhs.w; return lhs;
    }

    // component-wise vector subtract assign
    CPU_AND_GPU friend Vector4<T> &operator -= (Vector4<T> &lhs, const Vector4<T> &rhs)	{
        lhs.x -= rhs.x; lhs.y -= rhs.y; lhs.z -= rhs.z; lhs.w -= rhs.w; return lhs;
    }

    // unary negate
    CPU_AND_GPU friend Vector4<T> operator - (const Vector4<T> &rhs)	{
        Vector4<T> rv; rv.x = -rhs.x; rv.y = -rhs.y; rv.z = -rhs.z; rv.w = -rhs.w; return rv;
    }

    // vector add
    CPU_AND_GPU friend Vector4<T> operator + (const Vector4<T> &lhs, const Vector4<T> &rhs) {
        Vector4<T> rv(lhs); return rv += rhs;
    }

    // vector subtract
    CPU_AND_GPU friend Vector4<T> operator - (const Vector4<T> &lhs, const Vector4<T> &rhs) {
        Vector4<T> rv(lhs); return rv -= rhs;
    }

    // scalar multiply
    CPU_AND_GPU friend Vector4<T> operator * (const Vector4<T> &lhs, T rhs) {
        Vector4<T> rv(lhs); return rv *= rhs;
    }

    // scalar multiply
    CPU_AND_GPU friend Vector4<T> operator * (T lhs, const Vector4<T> &rhs) {
        Vector4<T> rv(lhs); return rv *= rhs;
    }

    // vector component-wise multiply
    CPU_AND_GPU friend Vector4<T> operator * (const Vector4<T> &lhs, const Vector4<T> &rhs) {
        Vector4<T> rv(lhs); return rv *= rhs;
    }

    // scalar divide
    CPU_AND_GPU friend Vector4<T> operator / (const Vector4<T> &lhs, T rhs) {
        Vector4<T> rv(lhs); return rv /= rhs;
    }

    // vector component-wise divide
    CPU_AND_GPU friend Vector4<T> operator / (const Vector4<T> &lhs, const Vector4<T> &rhs) {
        Vector4<T> rv(lhs); return rv /= rhs;
    }

    ////////////////////////////////////////////////////////
    //  Comparison operators
    ////////////////////////////////////////////////////////

    // equality
    CPU_AND_GPU friend bool operator == (const Vector4<T> &lhs, const Vector4<T> &rhs) {
        return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z) && (lhs.w == rhs.w);
    }

    // inequality
    CPU_AND_GPU friend bool operator != (const Vector4<T> &lhs, const Vector4<T> &rhs) {
        return (lhs.x != rhs.x) || (lhs.y != rhs.y) || (lhs.z != rhs.z) || (lhs.w != rhs.w);
    }

    friend std::ostream& operator<<(std::ostream& os, const Vector4<T>& dt){
        os << dt.x << ", " << dt.y << ", " << dt.z << ", " << dt.w;
        return os;
    }
};

template <class T> class Vector6 : public Vector6_ < T >
{
public:
    typedef T value_type;
    CPU_AND_GPU inline int size() const { return 6; }

    ////////////////////////////////////////////////////////
    //  Constructors
    ////////////////////////////////////////////////////////

    CPU_AND_GPU Vector6() {} // Default constructor
    CPU_AND_GPU Vector6(const T &t) { this->v[0] = t; this->v[1] = t; this->v[2] = t; this->v[3] = t; this->v[4] = t; this->v[5] = t; } //Scalar constructor
    CPU_AND_GPU Vector6(const T *tp) { this->v[0] = tp[0]; this->v[1] = tp[1]; this->v[2] = tp[2]; this->v[3] = tp[3]; this->v[4] = tp[4]; this->v[5] = tp[5]; } // Construct from array
    CPU_AND_GPU Vector6(const T v0, const T v1, const T v2, const T v3, const T v4, const T v5) { this->v[0] = v0; this->v[1] = v1; this->v[2] = v2; this->v[3] = v3; this->v[4] = v4; this->v[5] = v5; } // Construct from explicit values
    CPU_AND_GPU explicit Vector6(const Vector4_<T> &u, T v0, T v1) { this->v[0] = u.x; this->v[1] = u.y; this->v[2] = u.z; this->v[3] = u.w; this->v[4] = v0; this->v[5] = v1; }
    CPU_AND_GPU explicit Vector6(const Vector3_<T> &u, T v0, T v1, T v2) { this->v[0] = u.x; this->v[1] = u.y; this->v[2] = u.z; this->v[3] = v0; this->v[4] = v1; this->v[5] = v2; }
    CPU_AND_GPU explicit Vector6(const Vector2_<T> &u, T v0, T v1, T v2, T v3) { this->v[0] = u.x; this->v[1] = u.y; this->v[2] = v0; this->v[3] = v1; this->v[4] = v2, this->v[5] = v3; }

    CPU_AND_GPU inline Vector6<int> toIntRound() const {
        return Vector6<int>((int)ROUND(this[0]), (int)ROUND(this[1]), (int)ROUND(this[2]), (int)ROUND(this[3]), (int)ROUND(this[4]), (int)ROUND(this[5]));
    }

    CPU_AND_GPU inline Vector6<unsigned char> toUChar() const {
        Vector6<int> vi = toIntRound(); return Vector6<unsigned char>((unsigned char)CLAMP(vi[0], 0, 255), (unsigned char)CLAMP(vi[1], 0, 255), (unsigned char)CLAMP(vi[2], 0, 255), (unsigned char)CLAMP(vi[3], 0, 255), (unsigned char)CLAMP(vi[4], 0, 255), (unsigned char)CLAMP(vi[5], 0, 255));
    }

    CPU_AND_GPU inline Vector6<float> toFloat() const {
        return Vector6<float>((float)this[0], (float)this[1], (float)this[2], (float)this[3], (float)this[4], (float)this[5]);
    }

    CPU_AND_GPU const T *getValues() const { return this->v; }
    CPU_AND_GPU Vector6<T> &setValues(const T *rhs) { this[0] = rhs[0]; this[1] = rhs[1]; this[2] = rhs[2]; this[3] = rhs[3]; this[4] = rhs[4]; this[5] = rhs[5]; return *this; }

    // indexing operators
    CPU_AND_GPU T &operator [](int i) { return this->v[i]; }
    CPU_AND_GPU const T &operator [](int i) const { return this->v[i]; }

    // type-cast operators
    CPU_AND_GPU operator T *() { return this->v; }
    CPU_AND_GPU operator const T *() const { return this->v; }

    ////////////////////////////////////////////////////////
    //  Math operators
    ////////////////////////////////////////////////////////

    // scalar multiply assign
    CPU_AND_GPU friend Vector6<T> &operator *= (Vector6<T> &lhs, T d) {
        lhs[0] *= d; lhs[1] *= d; lhs[2] *= d; lhs[3] *= d; lhs[4] *= d; lhs[5] *= d; return lhs;
    }

    // component-wise vector multiply assign
    CPU_AND_GPU friend Vector6<T> &operator *= (Vector6<T> &lhs, const Vector6<T> &rhs) {
        lhs[0] *= rhs[0]; lhs[1] *= rhs[1]; lhs[2] *= rhs[2]; lhs[3] *= rhs[3]; lhs[4] *= rhs[4]; lhs[5] *= rhs[5]; return lhs;
    }

    // scalar divide assign
    CPU_AND_GPU friend Vector6<T> &operator /= (Vector6<T> &lhs, T d){
        lhs[0] /= d; lhs[1] /= d; lhs[2] /= d; lhs[3] /= d; lhs[4] /= d; lhs[5] /= d; return lhs;
    }

    // component-wise vector divide assign
    CPU_AND_GPU friend Vector6<T> &operator /= (Vector6<T> &lhs, const Vector6<T> &rhs) {
        lhs[0] /= rhs[0]; lhs[1] /= rhs[1]; lhs[2] /= rhs[2]; lhs[3] /= rhs[3]; lhs[4] /= rhs[4]; lhs[5] /= rhs[5]; return lhs;
    }

    // component-wise vector add assign
    CPU_AND_GPU friend Vector6<T> &operator += (Vector6<T> &lhs, const Vector6<T> &rhs)	{
        lhs[0] += rhs[0]; lhs[1] += rhs[1]; lhs[2] += rhs[2]; lhs[3] += rhs[3]; lhs[4] += rhs[4]; lhs[5] += rhs[5]; return lhs;
    }

    // component-wise vector subtract assign
    CPU_AND_GPU friend Vector6<T> &operator -= (Vector6<T> &lhs, const Vector6<T> &rhs)	{
        lhs[0] -= rhs[0]; lhs[1] -= rhs[1]; lhs[2] -= rhs[2]; lhs[3] -= rhs[3]; lhs[4] -= rhs[4]; lhs[5] -= rhs[5];  return lhs;
    }

    // unary negate
    CPU_AND_GPU friend Vector6<T> operator - (const Vector6<T> &rhs)	{
        Vector6<T> rv; rv[0] = -rhs[0]; rv[1] = -rhs[1]; rv[2] = -rhs[2]; rv[3] = -rhs[3]; rv[4] = -rhs[4]; rv[5] = -rhs[5];  return rv;
    }

    // vector add
    CPU_AND_GPU friend Vector6<T> operator + (const Vector6<T> &lhs, const Vector6<T> &rhs) {
        Vector6<T> rv(lhs); return rv += rhs;
    }

    // vector subtract
    CPU_AND_GPU friend Vector6<T> operator - (const Vector6<T> &lhs, const Vector6<T> &rhs) {
        Vector6<T> rv(lhs); return rv -= rhs;
    }

    // scalar multiply
    CPU_AND_GPU friend Vector6<T> operator * (const Vector6<T> &lhs, T rhs) {
        Vector6<T> rv(lhs); return rv *= rhs;
    }

    // scalar multiply
    CPU_AND_GPU friend Vector6<T> operator * (T lhs, const Vector6<T> &rhs) {
        Vector6<T> rv(lhs); return rv *= rhs;
    }

    // vector component-wise multiply
    CPU_AND_GPU friend Vector6<T> operator * (const Vector6<T> &lhs, const Vector6<T> &rhs) {
        Vector6<T> rv(lhs); return rv *= rhs;
    }

    // scalar divide
    CPU_AND_GPU friend Vector6<T> operator / (const Vector6<T> &lhs, T rhs) {
        Vector6<T> rv(lhs); return rv /= rhs;
    }

    // vector component-wise divide
    CPU_AND_GPU friend Vector6<T> operator / (const Vector6<T> &lhs, const Vector6<T> &rhs) {
        Vector6<T> rv(lhs); return rv /= rhs;
    }

    ////////////////////////////////////////////////////////
    //  Comparison operators
    ////////////////////////////////////////////////////////

    // equality
    CPU_AND_GPU friend bool operator == (const Vector6<T> &lhs, const Vector6<T> &rhs) {
        return (lhs[0] == rhs[0]) && (lhs[1] == rhs[1]) && (lhs[2] == rhs[2]) && (lhs[3] == rhs[3]) && (lhs[4] == rhs[4]) && (lhs[5] == rhs[5]);
    }

    // inequality
    CPU_AND_GPU friend bool operator != (const Vector6<T> &lhs, const Vector6<T> &rhs) {
        return (lhs[0] != rhs[0]) || (lhs[1] != rhs[1]) || (lhs[2] != rhs[2]) || (lhs[3] != rhs[3]) || (lhs[4] != rhs[4]) || (lhs[5] != rhs[5]);
    }

    friend std::ostream& operator<<(std::ostream& os, const Vector6<T>& dt){
        os << dt[0] << ", " << dt[1] << ", " << dt[2] << ", " << dt[3] << ", " << dt[4] << ", " << dt[5];
        return os;
    }
};


template <class T, int s> class VectorX : public VectorX_ < T, s >
{
public:
    typedef T value_type;
    CPU_AND_GPU inline int size() const { return this->vsize; }

    ////////////////////////////////////////////////////////
    //  Constructors
    ////////////////////////////////////////////////////////

    CPU_AND_GPU VectorX() { this->vsize = s; } // Default constructor
    CPU_AND_GPU VectorX(const T &t) { for (int i = 0; i < s; i++) this->v[i] = t; } //Scalar constructor
    CPU_AND_GPU VectorX(const T *tp) { for (int i = 0; i < s; i++) this->v[i] = tp[i]; } // Construct from array

    // indexing operators
    CPU_AND_GPU T &operator [](int i) { return this->v[i]; }
    CPU_AND_GPU const T &operator [](int i) const { return this->v[i]; }


    CPU_AND_GPU inline VectorX<int, s> toIntRound() const {
        VectorX<int, s> retv;
        for (int i = 0; i < s; i++) retv[i] = (int)ROUND(this->v[i]);
        return retv;
    }

    CPU_AND_GPU inline VectorX<unsigned char, s> toUChar() const {
        VectorX<int, s> vi = toIntRound();
        VectorX<unsigned char, s> retv;
        for (int i = 0; i < s; i++) retv[i] = (unsigned char)CLAMP(vi[0], 0, 255);
        return retv;
    }

    CPU_AND_GPU inline VectorX<float, s> toFloat() const {
        VectorX<float, s> retv;
        for (int i = 0; i < s; i++) retv[i] = (float) this->v[i];
        return retv;
    }

    CPU_AND_GPU const T *getValues() const { return this->v; }
    CPU_AND_GPU VectorX<T, s> &setValues(const T *rhs) { for (int i = 0; i < s; i++) this->v[i] = rhs[i]; return *this; }
    CPU_AND_GPU void Clear(T v){
        for (int i = 0; i < s; i++)
            this->v[i] = v;
    }

    CPU_AND_GPU void setZeros(){
        Clear(0);
    }

    // type-cast operators
    CPU_AND_GPU operator T *() { return this->v; }
    CPU_AND_GPU operator const T *() const { return this->v; }

    ////////////////////////////////////////////////////////
    //  Math operators
    ////////////////////////////////////////////////////////

    // scalar multiply assign
    CPU_AND_GPU friend VectorX<T, s> &operator *= (VectorX<T, s> &lhs, T d) {
        for (int i = 0; i < s; i++) lhs[i] *= d; return lhs;
    }

    // component-wise vector multiply assign
    CPU_AND_GPU friend VectorX<T, s> &operator *= (VectorX<T, s> &lhs, const VectorX<T, s> &rhs) {
        for (int i = 0; i < s; i++) lhs[i] *= rhs[i]; return lhs;
    }

    // scalar divide assign
    CPU_AND_GPU friend VectorX<T, s> &operator /= (VectorX<T, s> &lhs, T d){
        for (int i = 0; i < s; i++) lhs[i] /= d; return lhs;
    }

    // component-wise vector divide assign
    CPU_AND_GPU friend VectorX<T, s> &operator /= (VectorX<T, s> &lhs, const VectorX<T, s> &rhs) {
        for (int i = 0; i < s; i++) lhs[i] /= rhs[i]; return lhs;
    }

    // component-wise vector add assign
    CPU_AND_GPU friend VectorX<T, s> &operator += (VectorX<T, s> &lhs, const VectorX<T, s> &rhs)	{
        for (int i = 0; i < s; i++) lhs[i] += rhs[i]; return lhs;
    }

    // component-wise vector subtract assign
    CPU_AND_GPU friend VectorX<T, s> &operator -= (VectorX<T, s> &lhs, const VectorX<T, s> &rhs)	{
        for (int i = 0; i < s; i++) lhs[i] -= rhs[i]; return lhs;
    }

    // unary negate
    CPU_AND_GPU friend VectorX<T, s> operator - (const VectorX<T, s> &rhs)	{
        VectorX<T, s> rv; for (int i = 0; i < s; i++) rv[i] = -rhs[i]; return rv;
    }

    // vector add
    CPU_AND_GPU friend VectorX<T, s> operator + (const VectorX<T, s> &lhs, const VectorX<T, s> &rhs) {
        VectorX<T, s> rv(lhs); return rv += rhs;
    }

    // vector subtract
    CPU_AND_GPU friend VectorX<T, s> operator - (const VectorX<T, s> &lhs, const VectorX<T, s> &rhs) {
        VectorX<T, s> rv(lhs); return rv -= rhs;
    }

    // scalar multiply
    CPU_AND_GPU friend VectorX<T, s> operator * (const VectorX<T, s> &lhs, T rhs) {
        VectorX<T, s> rv(lhs); return rv *= rhs;
    }

    // scalar multiply
    CPU_AND_GPU friend VectorX<T, s> operator * (T lhs, const VectorX<T, s> &rhs) {
        VectorX<T, s> rv(lhs); return rv *= rhs;
    }

    // vector component-wise multiply
    CPU_AND_GPU friend VectorX<T, s> operator * (const VectorX<T, s> &lhs, const VectorX<T, s> &rhs) {
        VectorX<T, s> rv(lhs); return rv *= rhs;
    }

    // scalar divide
    CPU_AND_GPU friend VectorX<T, s> operator / (const VectorX<T, s> &lhs, T rhs) {
        VectorX<T, s> rv(lhs); return rv /= rhs;
    }

    // vector component-wise divide
    CPU_AND_GPU friend VectorX<T, s> operator / (const VectorX<T, s> &lhs, const VectorX<T, s> &rhs) {
        VectorX<T, s> rv(lhs); return rv /= rhs;
    }

    ////////////////////////////////////////////////////////
    //  Comparison operators
    ////////////////////////////////////////////////////////

    // equality
    CPU_AND_GPU friend bool operator == (const VectorX<T, s> &lhs, const VectorX<T, s> &rhs) {
        for (int i = 0; i < s; i++) if (lhs[i] != rhs[i]) return false;
        return true;
    }

    // inequality
    CPU_AND_GPU friend bool operator != (const VectorX<T, s> &lhs, const Vector6<T> &rhs) {
        for (int i = 0; i < s; i++) if (lhs[i] != rhs[i]) return true;
        return false;
    }

    friend std::ostream& operator<<(std::ostream& os, const VectorX<T, s>& dt){
        for (int i = 0; i < s; i++) os << dt[i] << "\n";
        return os;
    }
};

////////////////////////////////////////////////////////////////////////////////
// Generic vector operations
////////////////////////////////////////////////////////////////////////////////

template< class T> CPU_AND_GPU inline T sqr(const T &v) { return v*v; }

// compute the dot product of two vectors
template<class T> CPU_AND_GPU inline typename T::value_type dot(const T &lhs, const T &rhs) {
    typename T::value_type r = 0;
    for (int i = 0; i < lhs.size(); i++)
        r += lhs[i] * rhs[i];
    return r;
}

// return the squared length of the provided vector, i.e. the dot product with itself
template< class T> CPU_AND_GPU inline typename T::value_type length2(const T &vec) {
    return dot(vec, vec);
}

// return the length of the provided vector
template< class T> CPU_AND_GPU inline typename T::value_type length(const T &vec) {
    return sqrt(length2(vec));
}

// return the normalized version of the vector
template< class T> CPU_AND_GPU inline T normalize(const T &vec)	{
    typename T::value_type sum = length(vec);
    return sum == 0 ? T(typename T::value_type(0)) : vec / sum;
}

//template< class T> CPU_AND_GPU inline T min(const T &lhs, const T &rhs) {
//	return lhs <= rhs ? lhs : rhs;
//}

//template< class T> CPU_AND_GPU inline T max(const T &lhs, const T &rhs) {
//	return lhs >= rhs ? lhs : rhs;
//}

//component wise min
template< class T> CPU_AND_GPU inline T minV(const T &lhs, const T &rhs) {
    T rv;
    for (int i = 0; i < lhs.size(); i++)
        rv[i] = min(lhs[i], rhs[i]);
    return rv;
}

// component wise max
template< class T>
CPU_AND_GPU inline T maxV(const T &lhs, const T &rhs)	{
    T rv;
    for (int i = 0; i < lhs.size(); i++)
        rv[i] = max(lhs[i], rhs[i]);
    return rv;
}


// cross product
template< class T>
CPU_AND_GPU Vector3<T> cross(const Vector3<T> &lhs, const Vector3<T> &rhs) {
    Vector3<T> r;
    r.x = lhs.y * rhs.z - lhs.z * rhs.y;
    r.y = lhs.z * rhs.x - lhs.x * rhs.z;
    r.z = lhs.x * rhs.y - lhs.y * rhs.x;
    return r;
}























/************************************************************************/
/* WARNING: the following 3x3 and 4x4 matrix are using column major, to	*/
/* be consistent with OpenGL default rather than most C/C++ default.	*/
/* In all other parts of the code, we still use row major order.		*/
/************************************************************************/
template <class T> class Vector2;
template <class T> class Vector3;
template <class T> class Vector4;
template <class T, int s> class VectorX;

//////////////////////////////////////////////////////////////////////////
//						Basic Matrix Structure
//////////////////////////////////////////////////////////////////////////

template <class T> struct Matrix4_{
    union {
        struct { // Warning: see the header in this file for the special matrix order
            T m00, m01, m02, m03;	// |0, 4, 8,  12|    |m00, m10, m20, m30|
            T m10, m11, m12, m13;	// |1, 5, 9,  13|    |m01, m11, m21, m31|
            T m20, m21, m22, m23;	// |2, 6, 10, 14|    |m02, m12, m22, m32|
            T m30, m31, m32, m33;	// |3, 7, 11, 15|    |m03, m13, m23, m33|
        };
        T m[16];
    };
};

template <class T> struct Matrix3_{
    union { // Warning: see the header in this file for the special matrix order
        struct {
            T m00, m01, m02; // |0, 3, 6|     |m00, m10, m20|
            T m10, m11, m12; // |1, 4, 7|     |m01, m11, m21|
            T m20, m21, m22; // |2, 5, 8|     |m02, m12, m22|
        };
        T m[9];
    };
};

template<class T, int s> struct MatrixSQX_{
    int dim;
    int sq;
    T m[s*s];
};

template<class T>
class Matrix3;
//////////////////////////////////////////////////////////////////////////
// Matrix class with math operators
//////////////////////////////////////////////////////////////////////////
template<class T>
class Matrix4 : public Matrix4_ < T >
{
public:
    CPU_AND_GPU Matrix4() {}
    CPU_AND_GPU Matrix4(T t) { setValues(t); }
    CPU_AND_GPU Matrix4(const T *m)	{ setValues(m); }
    CPU_AND_GPU Matrix4(T a00, T a01, T a02, T a03, T a10, T a11, T a12, T a13, T a20, T a21, T a22, T a23, T a30, T a31, T a32, T a33)	{
        this->m00 = a00; this->m01 = a01; this->m02 = a02; this->m03 = a03;
        this->m10 = a10; this->m11 = a11; this->m12 = a12; this->m13 = a13;
        this->m20 = a20; this->m21 = a21; this->m22 = a22; this->m23 = a23;
        this->m30 = a30; this->m31 = a31; this->m32 = a32; this->m33 = a33;
    }

#define Rij(row, col) R.m[row + 3 * col]
    CPU_AND_GPU Matrix3<T> GetR(void) const
    {
        Matrix3<T> R;
        Rij(0, 0) = m[0 + 4 * 0]; Rij(1, 0) = m[1 + 4 * 0]; Rij(2, 0) = m[2 + 4 * 0];
        Rij(0, 1) = m[0 + 4 * 1]; Rij(1, 1) = m[1 + 4 * 1]; Rij(2, 1) = m[2 + 4 * 1];
        Rij(0, 2) = m[0 + 4 * 2]; Rij(1, 2) = m[1 + 4 * 2]; Rij(2, 2) = m[2 + 4 * 2];

        return R;
    }

    CPU_AND_GPU void SetR(const Matrix3<T>& R) {
        m[0 + 4 * 0] = Rij(0, 0); m[1 + 4 * 0] = Rij(1, 0); m[2 + 4 * 0] = Rij(2, 0);
        m[0 + 4 * 1] = Rij(0, 1); m[1 + 4 * 1] = Rij(1, 1); m[2 + 4 * 1] = Rij(2, 1);
        m[0 + 4 * 2] = Rij(0, 2); m[1 + 4 * 2] = Rij(1, 2); m[2 + 4 * 2] = Rij(2, 2);
    }
#undef Rij

    CPU_AND_GPU inline void getValues(T *mp) const	{ memcpy(mp, this->m, sizeof(T) * 16); }
    CPU_AND_GPU inline const T *getValues() const { return this->m; }
    CPU_AND_GPU inline Vector3<T> getScale() const { return Vector3<T>(this->m00, this->m11, this->m22); }

    // Element access
    CPU_AND_GPU inline T &operator()(int x, int y)	{ return at(x, y); }
    CPU_AND_GPU inline const T &operator()(int x, int y) const	{ return at(x, y); }
    CPU_AND_GPU inline T &operator()(Vector2<int> pnt)	{ return at(pnt.x, pnt.y); }
    CPU_AND_GPU inline const T &operator()(Vector2<int> pnt) const	{ return at(pnt.x, pnt.y); }
    CPU_AND_GPU inline T &at(int x, int y) { return this->m[y | (x << 2)]; }
    CPU_AND_GPU inline const T &at(int x, int y) const { return this->m[y | (x << 2)]; }

    // set values
    CPU_AND_GPU inline void setValues(const T *mp) { memcpy(this->m, mp, sizeof(T) * 16); }
    CPU_AND_GPU inline void setValues(T r)	{ for (int i = 0; i < 16; i++)	this->m[i] = r; }
    CPU_AND_GPU inline void setZeros() { memset(this->m, 0, sizeof(T) * 16); }
    CPU_AND_GPU inline void setIdentity() { setZeros(); this->m00 = this->m11 = this->m22 = this->m33 = 1; }
    CPU_AND_GPU inline void setScale(T s) { this->m00 = this->m11 = this->m22 = s; }
    CPU_AND_GPU inline void setScale(const Vector3_<T> &s) { this->m00 = s.v[0]; this->m11 = s.v[1]; this->m22 = s.v[2]; }
    CPU_AND_GPU inline void setTranslate(const Vector3_<T> &t) { for (int y = 0; y < 3; y++) at(3, y) = t.v[y]; }
    CPU_AND_GPU inline void setRow(int r, const Vector4_<T> &t){ for (int x = 0; x < 4; x++) at(x, r) = t.v[x]; }
    CPU_AND_GPU inline void setColumn(int c, const Vector4_<T> &t) { memcpy(this->m + 4 * c, t.v, sizeof(T) * 4); }

    // get values
    CPU_AND_GPU inline Vector3<T> getTranslate() const {
        Vector3<T> T;
        for (int y = 0; y < 3; y++)
            T.v[y] = m[y + 4 * 3];
        return T;
    }
    CPU_AND_GPU inline Vector4<T> getRow(int r) const { Vector4<T> v; for (int x = 0; x < 4; x++) v.v[x] = at(x, r); return v; }
    CPU_AND_GPU inline Vector4<T> getColumn(int c) const { Vector4<T> v; memcpy(v.v, this->m + 4 * c, sizeof(T) * 4); return v; }
    CPU_AND_GPU inline Matrix4 t() { // transpose
        Matrix4 mtrans;
        for (int x = 0; x < 4; x++)	for (int y = 0; y < 4; y++)
            mtrans(x, y) = at(y, x);
        return mtrans;
    }

    CPU_AND_GPU inline friend Matrix4 operator * (const Matrix4 &lhs, const Matrix4 &rhs)	{
        Matrix4 r;
        r.setZeros();
        for (int x = 0; x < 4; x++) for (int y = 0; y < 4; y++) for (int k = 0; k < 4; k++)
            r(x, y) += lhs(k, y) * rhs(x, k);
        return r;
    }

    CPU_AND_GPU inline friend Matrix4 operator + (const Matrix4 &lhs, const Matrix4 &rhs) {
        Matrix4 res(lhs.m);
        return res += rhs;
    }

    CPU_AND_GPU inline Vector4<T> operator *(const Vector4<T> &rhs) const {
        Vector4<T> r;
        r[0] = this->m[0] * rhs[0] + this->m[4] * rhs[1] + this->m[8] * rhs[2] + this->m[12] * rhs[3];
        r[1] = this->m[1] * rhs[0] + this->m[5] * rhs[1] + this->m[9] * rhs[2] + this->m[13] * rhs[3];
        r[2] = this->m[2] * rhs[0] + this->m[6] * rhs[1] + this->m[10] * rhs[2] + this->m[14] * rhs[3];
        r[3] = this->m[3] * rhs[0] + this->m[7] * rhs[1] + this->m[11] * rhs[2] + this->m[15] * rhs[3];
        return r;
    }

    // Used as a projection matrix to multiply with the Vector3
    CPU_AND_GPU inline Vector3<T> operator *(const Vector3<T> &rhs) const {
        Vector3<T> r;
        r[0] = this->m[0] * rhs[0] + this->m[4] * rhs[1] + this->m[8] * rhs[2] + this->m[12];
        r[1] = this->m[1] * rhs[0] + this->m[5] * rhs[1] + this->m[9] * rhs[2] + this->m[13];
        r[2] = this->m[2] * rhs[0] + this->m[6] * rhs[1] + this->m[10] * rhs[2] + this->m[14];
        return r;
    }

    CPU_AND_GPU inline friend Vector4<T> operator *(const Vector4<T> &lhs, const Matrix4 &rhs){
        Vector4<T> r;
        for (int x = 0; x < 4; x++)
            r[x] = lhs[0] * rhs(x, 0) + lhs[1] * rhs(x, 1) + lhs[2] * rhs(x, 2) + lhs[3] * rhs(x, 3);
        return r;
    }

    CPU_AND_GPU inline Matrix4& operator += (const T &r) { for (int i = 0; i < 16; ++i) this->m[i] += r; return *this; }
    CPU_AND_GPU inline Matrix4& operator -= (const T &r) { for (int i = 0; i < 16; ++i) this->m[i] -= r; return *this; }
    CPU_AND_GPU inline Matrix4& operator *= (const T &r) { for (int i = 0; i < 16; ++i) this->m[i] *= r; return *this; }
    CPU_AND_GPU inline Matrix4& operator /= (const T &r) { for (int i = 0; i < 16; ++i) this->m[i] /= r; return *this; }
    CPU_AND_GPU inline Matrix4 &operator += (const Matrix4 &mat) { for (int i = 0; i < 16; ++i) this->m[i] += mat.m[i]; return *this; }
    CPU_AND_GPU inline Matrix4 &operator -= (const Matrix4 &mat) { for (int i = 0; i < 16; ++i) this->m[i] -= mat.m[i]; return *this; }

    CPU_AND_GPU inline friend bool operator == (const Matrix4 &lhs, const Matrix4 &rhs) {
        bool r = lhs.m[0] == rhs.m[0];
        for (int i = 1; i < 16; i++)
            r &= lhs.m[i] == rhs.m[i];
        return r;
    }

    CPU_AND_GPU inline friend bool operator != (const Matrix4 &lhs, const Matrix4 &rhs) {
        bool r = lhs.m[0] != rhs.m[0];
        for (int i = 1; i < 16; i++)
            r |= lhs.m[i] != rhs.m[i];
        return r;
    }

    CPU_AND_GPU inline Matrix4 getInv() const {
        Matrix4 out;
        this->inv(out);
        return out;
    }
    /// Set out to be the inverse matrix of this.
    CPU_AND_GPU inline bool inv(Matrix4 &out) const {
        T tmp[12], src[16], det;
        T *dst = out.m;
        for (int i = 0; i < 4; i++) {
            src[i] = this->m[i * 4];
            src[i + 4] = this->m[i * 4 + 1];
            src[i + 8] = this->m[i * 4 + 2];
            src[i + 12] = this->m[i * 4 + 3];
        }

        tmp[0] = src[10] * src[15];
        tmp[1] = src[11] * src[14];
        tmp[2] = src[9] * src[15];
        tmp[3] = src[11] * src[13];
        tmp[4] = src[9] * src[14];
        tmp[5] = src[10] * src[13];
        tmp[6] = src[8] * src[15];
        tmp[7] = src[11] * src[12];
        tmp[8] = src[8] * src[14];
        tmp[9] = src[10] * src[12];
        tmp[10] = src[8] * src[13];
        tmp[11] = src[9] * src[12];

        dst[0] = (tmp[0] * src[5] + tmp[3] * src[6] + tmp[4] * src[7]) - (tmp[1] * src[5] + tmp[2] * src[6] + tmp[5] * src[7]);
        dst[1] = (tmp[1] * src[4] + tmp[6] * src[6] + tmp[9] * src[7]) - (tmp[0] * src[4] + tmp[7] * src[6] + tmp[8] * src[7]);
        dst[2] = (tmp[2] * src[4] + tmp[7] * src[5] + tmp[10] * src[7]) - (tmp[3] * src[4] + tmp[6] * src[5] + tmp[11] * src[7]);
        dst[3] = (tmp[5] * src[4] + tmp[8] * src[5] + tmp[11] * src[6]) - (tmp[4] * src[4] + tmp[9] * src[5] + tmp[10] * src[6]);

        det = src[0] * dst[0] + src[1] * dst[1] + src[2] * dst[2] + src[3] * dst[3];
        if (det == 0.0f)
            return false;

        dst[4] = (tmp[1] * src[1] + tmp[2] * src[2] + tmp[5] * src[3]) - (tmp[0] * src[1] + tmp[3] * src[2] + tmp[4] * src[3]);
        dst[5] = (tmp[0] * src[0] + tmp[7] * src[2] + tmp[8] * src[3]) - (tmp[1] * src[0] + tmp[6] * src[2] + tmp[9] * src[3]);
        dst[6] = (tmp[3] * src[0] + tmp[6] * src[1] + tmp[11] * src[3]) - (tmp[2] * src[0] + tmp[7] * src[1] + tmp[10] * src[3]);
        dst[7] = (tmp[4] * src[0] + tmp[9] * src[1] + tmp[10] * src[2]) - (tmp[5] * src[0] + tmp[8] * src[1] + tmp[11] * src[2]);

        tmp[0] = src[2] * src[7];
        tmp[1] = src[3] * src[6];
        tmp[2] = src[1] * src[7];
        tmp[3] = src[3] * src[5];
        tmp[4] = src[1] * src[6];
        tmp[5] = src[2] * src[5];
        tmp[6] = src[0] * src[7];
        tmp[7] = src[3] * src[4];
        tmp[8] = src[0] * src[6];
        tmp[9] = src[2] * src[4];
        tmp[10] = src[0] * src[5];
        tmp[11] = src[1] * src[4];

        dst[8] = (tmp[0] * src[13] + tmp[3] * src[14] + tmp[4] * src[15]) - (tmp[1] * src[13] + tmp[2] * src[14] + tmp[5] * src[15]);
        dst[9] = (tmp[1] * src[12] + tmp[6] * src[14] + tmp[9] * src[15]) - (tmp[0] * src[12] + tmp[7] * src[14] + tmp[8] * src[15]);
        dst[10] = (tmp[2] * src[12] + tmp[7] * src[13] + tmp[10] * src[15]) - (tmp[3] * src[12] + tmp[6] * src[13] + tmp[11] * src[15]);
        dst[11] = (tmp[5] * src[12] + tmp[8] * src[13] + tmp[11] * src[14]) - (tmp[4] * src[12] + tmp[9] * src[13] + tmp[10] * src[14]);
        dst[12] = (tmp[2] * src[10] + tmp[5] * src[11] + tmp[1] * src[9]) - (tmp[4] * src[11] + tmp[0] * src[9] + tmp[3] * src[10]);
        dst[13] = (tmp[8] * src[11] + tmp[0] * src[8] + tmp[7] * src[10]) - (tmp[6] * src[10] + tmp[9] * src[11] + tmp[1] * src[8]);
        dst[14] = (tmp[6] * src[9] + tmp[11] * src[11] + tmp[3] * src[8]) - (tmp[10] * src[11] + tmp[2] * src[8] + tmp[7] * src[9]);
        dst[15] = (tmp[10] * src[10] + tmp[4] * src[8] + tmp[9] * src[9]) - (tmp[8] * src[9] + tmp[11] * src[10] + tmp[5] * src[8]);

        out *= 1 / det;
        return true;
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix4<T>& dt) {
        for (int y = 0; y < 4; y++)
            os << dt(0, y) << ", " << dt(1, y) << ", " << dt(2, y) << ", " << dt(3, y) << "\n";
        return os;
    }
};

template<class T>
class Matrix3 : public Matrix3_ < T >
{
public:
    CPU_AND_GPU Matrix3() {}
    CPU_AND_GPU Matrix3(T t) { setValues(t); }
    CPU_AND_GPU Matrix3(const T *m)	{ setValues(m); }
    CPU_AND_GPU Matrix3(T a00, T a01, T a02, T a10, T a11, T a12, T a20, T a21, T a22)	{
        this->m00 = a00; this->m01 = a01; this->m02 = a02;
        this->m10 = a10; this->m11 = a11; this->m12 = a12;
        this->m20 = a20; this->m21 = a21; this->m22 = a22;
    }

    CPU_AND_GPU inline void getValues(T *mp) const	{ memcpy(mp, this->m, sizeof(T) * 9); }
    CPU_AND_GPU inline const T *getValues() const { return this->m; }
    CPU_AND_GPU inline Vector3<T> getScale() const { return Vector3<T>(this->m00, this->m11, this->m22); }

    // Element access
    CPU_AND_GPU inline T &operator()(int x, int y)	{ return at(x, y); }
    CPU_AND_GPU inline const T &operator()(int x, int y) const	{ return at(x, y); }
    CPU_AND_GPU inline T &operator()(Vector2<int> pnt)	{ return at(pnt.x, pnt.y); }
    CPU_AND_GPU inline const T &operator()(Vector2<int> pnt) const	{ return at(pnt.x, pnt.y); }
    CPU_AND_GPU inline T &at(int x, int y) { return this->m[x * 3 + y]; }
    CPU_AND_GPU inline const T &at(int x, int y) const { return this->m[x * 3 + y]; }

    // set values
    CPU_AND_GPU inline void setValues(const T *mp) { memcpy(this->m, mp, sizeof(T) * 9); }
    CPU_AND_GPU inline void setValues(const T r)	{ for (int i = 0; i < 9; i++)	this->m[i] = r; }
    CPU_AND_GPU inline void setZeros() { memset(this->m, 0, sizeof(T) * 9); }
    CPU_AND_GPU inline void setIdentity() { setZeros(); this->m00 = this->m11 = this->m22 = 1; }
    CPU_AND_GPU inline void setScale(T s) { this->m00 = this->m11 = this->m22 = s; }
    CPU_AND_GPU inline void setScale(const Vector3_<T> &s) { this->m00 = s[0]; this->m11 = s[1]; this->m22 = s[2]; }
    CPU_AND_GPU inline void setRow(int r, const Vector3_<T> &t){ for (int x = 0; x < 3; x++) at(x, r) = t[x]; }
    CPU_AND_GPU inline void setColumn(int c, const Vector3_<T> &t) { memcpy(this->m + 3 * c, t.v, sizeof(T) * 3); }

    // get values
    CPU_AND_GPU inline Vector3<T> getRow(int r) const { Vector3<T> v; for (int x = 0; x < 3; x++) v[x] = at(x, r); return v; }
    CPU_AND_GPU inline Vector3<T> getColumn(int c) const { Vector3<T> v; memcpy(v.v, this->m + 3 * c, sizeof(T) * 3); return v; }
    CPU_AND_GPU inline Matrix3 t() { // transpose
        Matrix3 mtrans;
        for (int x = 0; x < 3; x++)	for (int y = 0; y < 3; y++)
            mtrans(x, y) = at(y, x);
        return mtrans;
    }

    CPU_AND_GPU inline friend Matrix3 operator * (const Matrix3 &lhs, const Matrix3 &rhs)	{
        Matrix3 r;
        r.setZeros();
        for (int x = 0; x < 3; x++) for (int y = 0; y < 3; y++) for (int k = 0; k < 3; k++)
            r(x, y) += lhs(k, y) * rhs(x, k);
        return r;
    }

    CPU_AND_GPU inline friend Matrix3 operator + (const Matrix3 &lhs, const Matrix3 &rhs) {
        Matrix3 res(lhs.m);
        return res += rhs;
    }

    CPU_AND_GPU inline Vector3<T> operator *(const Vector3<T> &rhs) const {
        Vector3<T> r;
        r[0] = this->m[0] * rhs[0] + this->m[3] * rhs[1] + this->m[6] * rhs[2];
        r[1] = this->m[1] * rhs[0] + this->m[4] * rhs[1] + this->m[7] * rhs[2];
        r[2] = this->m[2] * rhs[0] + this->m[5] * rhs[1] + this->m[8] * rhs[2];
        return r;
    }

    CPU_AND_GPU inline Matrix3& operator *(const T &r) const {
        Matrix3 res(this->m);
        return res *= r;
    }

    CPU_AND_GPU inline friend Vector3<T> operator *(const Vector3<T> &lhs, const Matrix3 &rhs){
        Vector3<T> r;
        for (int x = 0; x < 3; x++)
            r[x] = lhs[0] * rhs(x, 0) + lhs[1] * rhs(x, 1) + lhs[2] * rhs(x, 2);
        return r;
    }

    CPU_AND_GPU inline Matrix3& operator += (const T &r) { for (int i = 0; i < 9; ++i) this->m[i] += r; return *this; }
    CPU_AND_GPU inline Matrix3& operator -= (const T &r) { for (int i = 0; i < 9; ++i) this->m[i] -= r; return *this; }
    CPU_AND_GPU inline Matrix3& operator *= (const T &r) { for (int i = 0; i < 9; ++i) this->m[i] *= r; return *this; }
    CPU_AND_GPU inline Matrix3& operator /= (const T &r) { for (int i = 0; i < 9; ++i) this->m[i] /= r; return *this; }
    CPU_AND_GPU inline Matrix3& operator += (const Matrix3 &mat) { for (int i = 0; i < 9; ++i) this->m[i] += mat.m[i]; return *this; }
    CPU_AND_GPU inline Matrix3& operator -= (const Matrix3 &mat) { for (int i = 0; i < 9; ++i) this->m[i] -= mat.m[i]; return *this; }

    CPU_AND_GPU inline friend bool operator == (const Matrix3 &lhs, const Matrix3 &rhs) {
        bool r = lhs[0] == rhs[0];
        for (int i = 1; i < 9; i++)
            r &= lhs[i] == rhs[i];
        return r;
    }

    CPU_AND_GPU inline friend bool operator != (const Matrix3 &lhs, const Matrix3 &rhs) {
        bool r = lhs[0] != rhs[0];
        for (int i = 1; i < 9; i++)
            r |= lhs[i] != rhs[i];
        return r;
    }

    /// Matrix determinant
    CPU_AND_GPU inline T det() const {
        return (this->m11*this->m22 - this->m12*this->m21)*this->m00 + (this->m12*this->m20 - this->m10*this->m22)*this->m01 + (this->m10*this->m21 - this->m11*this->m20)*this->m02;
    }

    /// The inverse matrix for float/double type
    CPU_AND_GPU inline bool inv(Matrix3 &out) const {
        T determinant = det();
        if (determinant == 0) {
            out.setZeros();
            return false;
        }

        out.m00 = (this->m11*this->m22 - this->m12*this->m21) / determinant;
        out.m01 = (this->m02*this->m21 - this->m01*this->m22) / determinant;
        out.m02 = (this->m01*this->m12 - this->m02*this->m11) / determinant;
        out.m10 = (this->m12*this->m20 - this->m10*this->m22) / determinant;
        out.m11 = (this->m00*this->m22 - this->m02*this->m20) / determinant;
        out.m12 = (this->m02*this->m10 - this->m00*this->m12) / determinant;
        out.m20 = (this->m10*this->m21 - this->m11*this->m20) / determinant;
        out.m21 = (this->m01*this->m20 - this->m00*this->m21) / determinant;
        out.m22 = (this->m00*this->m11 - this->m01*this->m10) / determinant;
        return true;
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix3<T>& dt)	{
        for (int y = 0; y < 3; y++)
            os << dt(0, y) << ", " << dt(1, y) << ", " << dt(2, y) << "\n";
        return os;
    }
};

template<class T, int s>
class MatrixSQX : public MatrixSQX_ < T, s >
{
public:
    CPU_AND_GPU MatrixSQX() { this->dim = s; this->sq = s*s; }
    CPU_AND_GPU MatrixSQX(T t) { this->dim = s; this->sq = s*s; setValues(t); }
    CPU_AND_GPU MatrixSQX(const T *m)	{ this->dim = s; this->sq = s*s; setValues(m); }
    CPU_AND_GPU MatrixSQX(const T m[s][s])	{ this->dim = s; this->sq = s*s; setValues((T*)m); }

    CPU_AND_GPU inline void getValues(T *mp) const	{ memcpy(mp, this->m, sizeof(T) * 16); }
    CPU_AND_GPU inline const T *getValues() const { return this->m; }

    CPU_AND_GPU static inline MatrixSQX<T, s> make_aaT(const VectorX<float, s>& a) {
        float a_aT[s][s];
        for (int c = 0; c < s; c++)
            for (int r = 0; r < s; r++)
                a_aT[c][r] = a[c] * a[r];
        return a_aT;
    }

    // Element access
    CPU_AND_GPU inline T &operator()(int x, int y)	{ return at(x, y); }
    CPU_AND_GPU inline const T &operator()(int x, int y) const	{ return at(x, y); }
    CPU_AND_GPU inline T &operator()(Vector2<int> pnt)	{ return at(pnt.x, pnt.y); }
    CPU_AND_GPU inline const T &operator()(Vector2<int> pnt) const	{ return at(pnt.x, pnt.y); }
    CPU_AND_GPU inline T &at(int x, int y) { return this->m[y * s + x]; }
    CPU_AND_GPU inline const T &at(int x, int y) const { return this->m[y * s + x]; }

    // set values
    CPU_AND_GPU inline void setValues(const T *mp) { for (int i = 0; i < s*s; i++) this->m[i] = mp[i]; }
    CPU_AND_GPU inline void setValues(T r)	{ for (int i = 0; i < s*s; i++)	this->m[i] = r; }
    CPU_AND_GPU inline void setZeros() { for (int i = 0; i < s*s; i++)	this->m[i] = 0; }
    CPU_AND_GPU inline void setIdentity() { setZeros(); for (int i = 0; i < s*s; i++) this->m[i + i*s] = 1; }

    // get values
    CPU_AND_GPU inline VectorX<T, s> getRow(int r) const { VectorX<T, s> v; for (int x = 0; x < s; x++) v[x] = at(x, r); return v; }
    CPU_AND_GPU inline VectorX<T, s> getColumn(int c) const { Vector4<T> v; for (int x = 0; x < s; x++) v[x] = at(c, x); return v; }
    CPU_AND_GPU inline MatrixSQX<T, s> getTranspose()
    { // transpose
        MatrixSQX<T, s> mtrans;
        for (int x = 0; x < s; x++)	for (int y = 0; y < s; y++)
            mtrans(x, y) = at(y, x);
        return mtrans;
    }

    CPU_AND_GPU inline friend  MatrixSQX<T, s> operator * (const  MatrixSQX<T, s> &lhs, const  MatrixSQX<T, s> &rhs)	{
        MatrixSQX<T, s> r;
        r.setZeros();
        for (int x = 0; x < s; x++) for (int y = 0; y < s; y++) for (int k = 0; k < s; k++)
            r(x, y) += lhs(k, y) * rhs(x, k);
        return r;
    }

    CPU_AND_GPU inline friend MatrixSQX<T, s> operator + (const MatrixSQX<T, s> &lhs, const MatrixSQX<T, s> &rhs) {
        MatrixSQX<T, s> res(lhs.m);
        return res += rhs;
    }

    CPU_AND_GPU inline MatrixSQX<T, s>& operator += (const T &r) { for (int i = 0; i < s*s; ++i) this->m[i] += r; return *this; }
    CPU_AND_GPU inline MatrixSQX<T, s>& operator -= (const T &r) { for (int i = 0; i < s*s; ++i) this->m[i] -= r; return *this; }
    CPU_AND_GPU inline MatrixSQX<T, s>& operator *= (const T &r) { for (int i = 0; i < s*s; ++i) this->m[i] *= r; return *this; }
    CPU_AND_GPU inline MatrixSQX<T, s>& operator /= (const T &r) { for (int i = 0; i < s*s; ++i) this->m[i] /= r; return *this; }
    CPU_AND_GPU inline MatrixSQX<T, s> &operator += (const MatrixSQX<T, s> &mat) { for (int i = 0; i < s*s; ++i) this->m[i] += mat.m[i]; return *this; }
    CPU_AND_GPU inline MatrixSQX<T, s> &operator -= (const MatrixSQX<T, s> &mat) { for (int i = 0; i < s*s; ++i) this->m[i] -= mat.m[i]; return *this; }

    CPU_AND_GPU inline friend bool operator == (const MatrixSQX<T, s> &lhs, const MatrixSQX<T, s> &rhs) {
        bool r = lhs[0] == rhs[0];
        for (int i = 1; i < s*s; i++)
            r &= lhs[i] == rhs[i];
        return r;
    }

    CPU_AND_GPU inline friend bool operator != (const MatrixSQX<T, s> &lhs, const MatrixSQX<T, s> &rhs) {
        bool r = lhs[0] != rhs[0];
        for (int i = 1; i < s*s; i++)
            r |= lhs[i] != rhs[i];
        return r;
    }

    friend std::ostream& operator<<(std::ostream& os, const MatrixSQX<T, s>& dt) {
        for (int y = 0; y < s; y++)
        {
            for (int x = 0; x < s; x++) os << dt(x, y) << "\t";
            os << "\n";
        }
        return os;
    }
};




























class Cholesky
{
private:
    std::vector<float> cholesky;
    int size, rank;

public:
    // Solve Ax = b for A symmetric positive-definite of size*size
    template<int m>
    static void solve(
        const MatrixSQX<float, m>& mat,
        const VectorX<float, m>&  b,
        VectorX<float, m>& result) {
        solve((const float*)mat.m, m, (const float*)b.v, result.v);
    }

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





















/// Change a generator to a transformation as expected by thrust
/// Turn ai into the matrix ai ai^T and bi into the vector ai * bi
template<typename Generator, int m>
struct GenerateToTransformation {
    CPU_AND_GPU thrust::pair< MatrixSQX<float, m>, VectorX<float, m> > operator()(int i) {

        VectorX<float, m> ai;
        float bi;
        Generator::generate(i, ai, bi);

        // Construct ai_aiT, ai_bi
        return{
            MatrixSQX<float, m>::make_aaT(ai),
            ai * bi
        };
    }
};

























template <int m>
CPU_AND_GPU thrust::pair<MatrixSQX<float, m>, VectorX<float, m>> operator+(
    const thrust::pair<MatrixSQX<float, m>, VectorX<float, m>> & l,
    const thrust::pair<MatrixSQX<float, m>, VectorX<float, m>> & r) {
    return{l.first + r.first, l.second + r.second};
}

/**
Build A^T A and A^T b where A is n x m and b has n elements.

Row i (0-based) of A and b[i] are generated by Constructor::generate(int i, VectorX<float, m> out_ai, float& out_bi).
*/
template<class Constructor, int m>
thrust::pair<MatrixSQX<float, m>, VectorX<float, m>> construct(int n) {
    typedef MatrixSQX<float, m> TAtA;
    typedef VectorX<float, m> TAtb;
    typedef thrust::pair<TAtA, TAtb> TAtA_Atb;
    assert(m < 100);
    ::n = n;

    TAtA zero_AtA; zero_AtA.setZeros();
    TAtb zero_Atb; zero_Atb.setZeros();

    const int min = 0, max = n-1;
    thrust::counting_iterator<int> first(min);
    auto last = first + (max - min) + 1;

    auto result = transform_reduce(
        first,
        last,
        GenerateToTransformation<Constructor, m>(),
        TAtA_Atb(zero_AtA, zero_Atb),
        thrust::plus<TAtA_Atb>());

    return result;
}


template<class Constructor, int m>
VectorX<float, m> constructAndSolve(int n) {
    auto result = construct<Constructor, m>(n);
    auto sum_AtA = result.first;
    auto sum_Atb = result.second;

    auto x = VectorX<float, m>();
    Cholesky::solve(sum_AtA, sum_Atb, x);
    return x;
}




































































































void assertApproxEqual(float a, float b, int considered_initial_bits = 20) {
    assert(considered_initial_bits > 8 + 1); // should consider at least sign and full exponent
    assert(considered_initial_bits <= 32);

    unsigned int ai = *(unsigned int*)&a;
    unsigned int bi = *(unsigned int*)&b;
    auto ait = ai >> (32 - considered_initial_bits);
    auto bit = bi >> (32 - considered_initial_bits);

    assert(ait == bit, "%f != %f, %x != %x, %x != %x",
        a, b, ai, bi, ait, bit
        );
}

struct Trafo {
    CPU_AND_GPU int operator()(int i) {
        return 2 * i;
    }
};

struct ConstructExampleEquation {
    template<int m>
    static GPU_ONLY void generate(const int i, VectorX<float, m>& out_ai, float& out_bi/*[1]*/) {
        for (int j = 0; j < m; j++) {
            out_ai[j] = 0;
            if (i == j || i == 0 || j == 0)
                out_ai[j] = 1;
        }
        out_bi = i + 1;
        printf("thread %d %d %d\n", xyz(threadIdx));
    }
};
int main(int argc, char** argv)
{
    const int min = 1, max = 6;
    thrust::counting_iterator<int> first(min);
    auto last = first + (max - min) + 1;
    int res = transform_reduce(first, last, Trafo(), 0, thrust::plus<int>());
    assert(res == 2 * (max * (max + 1)) / 2);

    const int m = 6;
    const int n = 2 * REDUCE_BLOCK_SIZE;
    auto x = constructAndSolve<ConstructExampleEquation, m>(n);
    float expect[m] = {258.164, -87.2215, -86.2215, -85.2215, -84.2215, -83.2215};
    for (int i = 0; i < m; i++)
        assertApproxEqual(x[i], expect[i]);

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(400, 300);
    glutCreateWindow("Hello World");
    auto e = glGetError();
    glGenTextures(-10, 0);
    auto e2 = glGetError();

    ((void)0);//LAUNCH_KERNEL((k<int, 5>), 1, 1);

    return 0;
}