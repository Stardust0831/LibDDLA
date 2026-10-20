// ---------------------------------------------------------------------------
// Backend-templated coverage of ddla::getrf, ddla::potrf and ddla::laswp -- the LAPACK-level wrappers that used to
// be raw cuSOLVER / hipSOLVER helpers in private headers (getrf.h, potrf.h, laswp.h) and now have a CPU arm
// (host LAPACK) and a GPU arm behind one `<Backend, T>` interface.
//
//   getrf  square / tall / wide shapes, padded leading dimension with the padding checked untouched, pivot range,
//          A = P*L*U rebuilt by hand and compared with the input, `info` for an exactly singular matrix;
//   potrf  'U' and 'L' on a Hermitian positive-definite matrix, A = U^H U / L L^H rebuilt by hand, the unreferenced
//          triangle checked bit for bit, `info` for a matrix that is not positive definite;
//   laswp  bit-exact against a hand-written LAPACK loop for incx = 1, 2, -1, -2 and sub-ranges k1..k2 (incx = 1 is
//          the vendor routine on CUDA, every other stride the row-by-row fallback), and incx = 1 followed by
//          incx = -1 restoring the matrix;
//   all    empty problems (info = 0) and the argument checks (null handle, backend mismatch, bad flag, null
//          pointers, leading dimension, negative sizes).
// Every pointer lives where the handle's backend does (host memory on a CPU handle, device memory on a GPU
// handle), `ipiv` and `info` included.  The reference arithmetic is written out by hand, not taken from the library.
//
// Follows the structure of test_backend_geam.cpp.  Debug aid:
//   DDLA_TEST_ONLY=cpu|gpu   run one backend alone
// ---------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

#include <ddla/ddla.h>

using namespace ddla;

namespace {

int g_rank = 0;

// -- deterministic data ------------------------------------------------------
struct Rng {
    std::uint64_t s;
    explicit Rng(std::uint64_t seed) : s(seed) {}
    std::uint64_t next()
    {
        s += 0x9E3779B97F4A7C15ull;
        std::uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uni() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0; }
};

template <typename T> struct Tr;
template <> struct Tr<float> {
    typedef float real;
    static float make(double a, double) { return static_cast<float>(a); }
    static float conjv(float x) { return x; }
    static double eps() { return 1.2e-7; }
    static const bool cplx = false;
};
template <> struct Tr<double> {
    typedef double real;
    static double make(double a, double) { return a; }
    static double conjv(double x) { return x; }
    static double eps() { return 2.3e-16; }
    static const bool cplx = false;
};
template <> struct Tr<std::complex<float>> {
    typedef float real;
    static std::complex<float> make(double a, double b) { return std::complex<float>(float(a), float(b)); }
    static std::complex<float> conjv(const std::complex<float>& x) { return std::conj(x); }
    static double eps() { return 1.2e-7; }
    static const bool cplx = true;
};
template <> struct Tr<std::complex<double>> {
    typedef double real;
    static std::complex<double> make(double a, double b) { return std::complex<double>(a, b); }
    static std::complex<double> conjv(const std::complex<double>& x) { return std::conj(x); }
    static double eps() { return 2.3e-16; }
    static const bool cplx = true;
};

template <typename T>
std::vector<T> random_vec(std::size_t n, std::uint64_t seed)
{
    Rng r(seed);
    std::vector<T> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double a = r.uni(), b = r.uni();
        v[i] = Tr<T>::make(a, b);
    }
    return v;
}

template <typename T>
bool same_bits(const T& a, const T& b)
{
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

// -- device (or host, on a CPU handle) buffers --------------------------------
template <typename T>
struct Buf {
    DdlaHandle_t h;
    T* p = nullptr;
    std::size_t n = 0;
    Buf(const DdlaHandle_t& handle, std::size_t count) : h(handle), n(count)
    {
        void* q = nullptr;
        if (ddlaMalloc(h, &q, std::max<std::size_t>(1, n) * sizeof(T)) != ddlaStatus_t::DDLA_STATUS_SUCCESS)
            throw std::runtime_error("test_backend_lapack: ddlaMalloc failed");
        p = static_cast<T*>(q);
    }
    ~Buf() { ddlaFree(h, p); }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    void up(const std::vector<T>& v)
    {
        if (!v.empty()) ddlaMemcpy(h, p, v.data(), v.size() * sizeof(T));
        ddlaSynchronize(h);
    }
    void down(std::vector<T>& v)
    {
        v.resize(n);
        if (n > 0) ddlaMemcpy(h, v.data(), p, n * sizeof(T));
        ddlaSynchronize(h);
    }
};

void report(const char* what, const std::string& detail)
{
    if (g_rank != 0) return;
    std::cerr << "  FAIL " << what << " " << detail << std::endl;
}

std::string dims(int m, int n, int extra = -1)
{
    std::string s = "m=" + std::to_string(m) + " n=" + std::to_string(n);
    if (extra >= 0) s += " " + std::to_string(extra);
    return s;
}

template <typename T>
T guard_value()
{
    return Tr<T>::make(777.0, -555.0);
}

// -- getrf --------------------------------------------------------------------------
// zero_col >= 0: that column of A is exactly zero, so U(zero_col, zero_col) is the first zero pivot.
template <DdlaBackend Backend, typename T>
int check_getrf(const DdlaHandle_t& h, int m, int n, int pad, int zero_col, std::uint64_t seed)
{
    const int lda = m + pad;
    const int k = std::min(m, n);
    std::vector<T> A0 = random_vec<T>(static_cast<std::size_t>(lda) * n, seed);
    for (int j = 0; j < n; ++j)
        for (int i = m; i < lda; ++i) A0[i + static_cast<std::size_t>(j) * lda] = guard_value<T>();
    if (zero_col >= 0)
        for (int i = 0; i < m; ++i) A0[i + static_cast<std::size_t>(zero_col) * lda] = Tr<T>::make(0.0, 0.0);

    Buf<T> dA(h, A0.size());
    Buf<int> dpiv(h, static_cast<std::size_t>(k)), dinfo(h, 1);
    dA.up(A0);
    dpiv.up(std::vector<int>(static_cast<std::size_t>(k), -1));
    dinfo.up(std::vector<int>(1, -77));

    getrf<Backend, T>(h, m, n, dA.p, lda, dpiv.p, dinfo.p);
    ddlaSynchronize(h);

    std::vector<T> F;
    std::vector<int> piv, info;
    dA.down(F);
    dpiv.down(piv);
    dinfo.down(info);

    int failures = 0;
    const int want_info = zero_col >= 0 ? zero_col + 1 : 0;
    if (info[0] != want_info) {
        report("getrf info", dims(m, n, zero_col) + " got " + std::to_string(info[0]) + " want " + std::to_string(want_info));
        ++failures;
    }
    for (int j = 0; j < n; ++j)
        for (int i = m; i < lda; ++i)
            if (!same_bits(F[i + static_cast<std::size_t>(j) * lda], guard_value<T>())) {
                report("getrf padding written", dims(m, n) + " at (" + std::to_string(i) + "," + std::to_string(j) + ")");
                return failures + 1;
            }
    if (zero_col >= 0) return failures;      // a zero pivot: the vendor routines may leave L undefined below it

    for (int i = 0; i < k; ++i)
        if (piv[i] < i + 1 || piv[i] > m) {
            report("getrf pivot range", dims(m, n) + " ipiv[" + std::to_string(i) + "]=" + std::to_string(piv[i]));
            return failures + 1;
        }

    // A = P * L * U: multiply the factors, then undo the interchanges from the last to the first.
    auto at = [&](int i, int j) { return F[i + static_cast<std::size_t>(j) * lda]; };
    std::vector<T> R(static_cast<std::size_t>(m) * n, Tr<T>::make(0.0, 0.0));
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            T s = Tr<T>::make(0.0, 0.0);
            const int tmax = std::min(std::min(i, j), k - 1);
            for (int t = 0; t <= tmax; ++t) s += (i == t ? Tr<T>::make(1.0, 0.0) : at(i, t)) * at(t, j);
            R[i + static_cast<std::size_t>(j) * m] = s;
        }
    for (int t = k - 1; t >= 0; --t) {
        const int p = piv[t] - 1;
        if (p != t)
            for (int j = 0; j < n; ++j) std::swap(R[t + static_cast<std::size_t>(j) * m], R[p + static_cast<std::size_t>(j) * m]);
    }
    double err = 0.0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            err = std::max(err, static_cast<double>(std::abs(R[i + static_cast<std::size_t>(j) * m] -
                                                              A0[i + static_cast<std::size_t>(j) * lda])));
    const double tol = 200.0 * Tr<T>::eps() * std::max(m, n);
    if (!(err <= tol)) {
        report("getrf residual", dims(m, n) + " err " + std::to_string(err) + " tol " + std::to_string(tol));
        ++failures;
    }
    return failures;
}

// -- potrf --------------------------------------------------------------------------
// bad_k >= 0: the matrix is the identity except A(bad_k, bad_k) = -1, so the leading minor of order bad_k + 1 fails.
template <DdlaBackend Backend, typename T>
int check_potrf(const DdlaHandle_t& h, char uplo, int n, int pad, int bad_k, std::uint64_t seed)
{
    const int lda = n + pad;
    const std::size_t N = static_cast<std::size_t>(n);
    // Hermitian positive definite: A = B*B^H + n*I (diagonal made exactly real).
    std::vector<T> B = random_vec<T>(N * n, seed);
    std::vector<T> A(static_cast<std::size_t>(lda) * n, guard_value<T>());
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            T s = Tr<T>::make(0.0, 0.0);
            for (int t = 0; t < n; ++t) s += B[i + t * N] * Tr<T>::conjv(B[j + t * N]);
            if (i == j) s = Tr<T>::make(static_cast<double>(std::real(s)) + n, 0.0);
            A[i + static_cast<std::size_t>(j) * lda] = s;
        }
    if (bad_k >= 0)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                A[i + static_cast<std::size_t>(j) * lda] = Tr<T>::make(i == j ? (i == bad_k ? -1.0 : 1.0) : 0.0, 0.0);
    // The triangle potrf must not touch gets a sentinel that would wreck the result if it were read.
    std::vector<T> A0 = A;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            if ((uplo == 'U' && i > j) || (uplo == 'L' && i < j))
                A0[i + static_cast<std::size_t>(j) * lda] = Tr<T>::make(777.0 + i, -555.0 - j);

    Buf<T> dA(h, A0.size());
    Buf<int> dinfo(h, 1);
    dA.up(A0);
    dinfo.up(std::vector<int>(1, -77));
    potrf<Backend, T>(h, uplo, n, dA.p, lda, dinfo.p);
    ddlaSynchronize(h);
    std::vector<T> F;
    std::vector<int> info;
    dA.down(F);
    dinfo.down(info);

    int failures = 0;
    const int want_info = bad_k >= 0 ? bad_k + 1 : 0;
    if (info[0] != want_info) {
        report("potrf info", std::string("uplo=") + uplo + " " + dims(n, n, bad_k) + " got " + std::to_string(info[0]) +
                                 " want " + std::to_string(want_info));
        ++failures;
    }
    // Unreferenced triangle and padding: bit for bit what went in.
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < lda; ++i) {
            const bool referenced = i < n && ((uplo == 'U' && i <= j) || (uplo == 'L' && i >= j));
            if (referenced) continue;
            if (!same_bits(F[i + static_cast<std::size_t>(j) * lda], A0[i + static_cast<std::size_t>(j) * lda])) {
                report("potrf wrote outside the triangle", std::string("uplo=") + uplo + " " + dims(n, n) + " at (" +
                                                               std::to_string(i) + "," + std::to_string(j) + ")");
                return failures + 1;
            }
        }
    if (bad_k >= 0) return failures;

    double err = 0.0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            if (!((uplo == 'U' && i <= j) || (uplo == 'L' && i >= j))) continue;
            T s = Tr<T>::make(0.0, 0.0);
            const int tmax = std::min(i, j);
            for (int t = 0; t <= tmax; ++t) {
                if (uplo == 'U')
                    s += Tr<T>::conjv(F[t + static_cast<std::size_t>(i) * lda]) * F[t + static_cast<std::size_t>(j) * lda];
                else
                    s += F[i + static_cast<std::size_t>(t) * lda] * Tr<T>::conjv(F[j + static_cast<std::size_t>(t) * lda]);
            }
            err = std::max(err, static_cast<double>(std::abs(s - A[i + static_cast<std::size_t>(j) * lda])));
        }
    const double tol = 200.0 * Tr<T>::eps() * n * n;
    if (!(err <= tol)) {
        report("potrf residual", std::string("uplo=") + uplo + " " + dims(n, n) + " err " + std::to_string(err) + " tol " +
                                     std::to_string(tol));
        ++failures;
    }
    return failures;
}

// -- laswp --------------------------------------------------------------------------
template <typename T>
void laswp_ref(std::vector<T>& A, int lda, int n, int k1, int k2, const std::vector<int>& ipiv, int incx)
{
    const int stride = std::abs(incx);
    const int first = incx > 0 ? k1 : k2;
    const int last = incx > 0 ? k2 : k1;
    const int step = incx > 0 ? 1 : -1;
    for (int i = first;; i += step) {
        const int p = ipiv[(k1 - 1) + static_cast<std::size_t>(i - k1) * stride];
        if (p != i)
            for (int j = 0; j < n; ++j)
                std::swap(A[(i - 1) + static_cast<std::size_t>(j) * lda], A[(p - 1) + static_cast<std::size_t>(j) * lda]);
        if (i == last) break;
    }
}

template <DdlaBackend Backend, typename T>
int check_laswp(const DdlaHandle_t& h, int m, int n, int pad, int k1, int k2, int incx, std::uint64_t seed)
{
    const int lda = m + pad;
    const int stride = std::abs(incx);
    std::vector<T> A0 = random_vec<T>(static_cast<std::size_t>(lda) * n, seed);
    for (int j = 0; j < n; ++j)
        for (int i = m; i < lda; ++i) A0[i + static_cast<std::size_t>(j) * lda] = guard_value<T>();
    Rng r(seed ^ 0xABCDEFull);
    std::vector<int> ipiv(static_cast<std::size_t>(k2) * stride);
    for (std::size_t i = 0; i < ipiv.size(); ++i) ipiv[i] = 1 + static_cast<int>(r.next() % static_cast<std::uint64_t>(m));

    std::vector<T> want = A0;
    laswp_ref(want, lda, n, k1, k2, ipiv, incx);

    Buf<T> dA(h, A0.size());
    Buf<int> dpiv(h, ipiv.size());
    dA.up(A0);
    dpiv.up(ipiv);
    laswp<Backend, T>(h, n, dA.p, lda, k1, k2, dpiv.p, incx);
    ddlaSynchronize(h);
    std::vector<T> got;
    dA.down(got);

    int failures = 0;
    for (std::size_t i = 0; i < got.size(); ++i)
        if (!same_bits(got[i], want[i])) {
            report("laswp", dims(m, n) + " k1=" + std::to_string(k1) + " k2=" + std::to_string(k2) + " incx=" +
                                std::to_string(incx) + " first mismatch at " + std::to_string(i));
            ++failures;
            break;
        }

    // incx and -incx are inverse permutations: applying both restores the matrix.
    if (incx == 1) {
        laswp<Backend, T>(h, n, dA.p, lda, k1, k2, dpiv.p, -1);
        ddlaSynchronize(h);
        dA.down(got);
        for (std::size_t i = 0; i < got.size(); ++i)
            if (!same_bits(got[i], A0[i])) {
                report("laswp round trip (incx=1 then -1)", dims(m, n) + " k1=" + std::to_string(k1) + " k2=" +
                                                                std::to_string(k2) + " first mismatch at " + std::to_string(i));
                ++failures;
                break;
            }
    }
    return failures;
}

// -- empty problems ---------------------------------------------------------------------
template <DdlaBackend Backend, typename T>
int check_empty(const DdlaHandle_t& h)
{
    int failures = 0;
    Buf<T> dA(h, 16);
    Buf<int> dpiv(h, 4), dinfo(h, 1);
    auto info_after = [&](const char* what, const std::function<void()>& call) {
        dinfo.up(std::vector<int>(1, -77));
        call();
        ddlaSynchronize(h);
        std::vector<int> info;
        dinfo.down(info);
        if (info[0] != 0) {
            report(what, "info not set to 0 for an empty problem");
            ++failures;
        }
    };
    info_after("getrf m=0", [&] { getrf<Backend, T>(h, 0, 3, dA.p, 1, dpiv.p, dinfo.p); });
    info_after("getrf n=0", [&] { getrf<Backend, T>(h, 3, 0, dA.p, 3, dpiv.p, dinfo.p); });
    info_after("potrf n=0", [&] { potrf<Backend, T>(h, 'L', 0, dA.p, 1, dinfo.p); });
    // laswp: nothing to do, nothing to touch (k1 > k2, n = 0), null pointers allowed.
    try {
        laswp<Backend, T>(h, 4, static_cast<T*>(nullptr), 4, 3, 2, nullptr, 1);
        laswp<Backend, T>(h, 0, static_cast<T*>(nullptr), 4, 1, 4, nullptr, 1);
    } catch (const std::exception& e) {
        report("laswp empty", e.what());
        ++failures;
    }
    return failures;
}

template <DdlaBackend Backend, typename T>
int run_type(const DdlaHandle_t& h)
{
    int failures = 0;
    struct Shape { int m, n; };
    const Shape shapes[] = {{1, 1}, {1, 7}, {7, 1}, {5, 5}, {37, 37}, {64, 64}, {50, 20}, {20, 50}, {130, 130}, {200, 150}};
    std::uint64_t seed = 1000;
    for (const Shape& s : shapes) {
        failures += check_getrf<Backend, T>(h, s.m, s.n, 0, -1, ++seed);
        failures += check_getrf<Backend, T>(h, s.m, s.n, 5, -1, ++seed);
    }
    failures += check_getrf<Backend, T>(h, 40, 40, 3, 17, ++seed);     // exactly singular: first zero pivot at 18
    failures += check_getrf<Backend, T>(h, 33, 33, 0, 0, ++seed);      // ... and at 1

    const int potrf_sizes[] = {1, 2, 5, 33, 64, 100};
    for (int n : potrf_sizes)
        for (char uplo : {'U', 'L'}) {
            failures += check_potrf<Backend, T>(h, uplo, n, 0, -1, ++seed);
            failures += check_potrf<Backend, T>(h, uplo, n, 4, -1, ++seed);
        }
    for (char uplo : {'U', 'L'}) {
        failures += check_potrf<Backend, T>(h, uplo, 8, 2, 3, ++seed);   // leading minor of order 4 fails
        failures += check_potrf<Backend, T>(h, uplo, 8, 0, 0, ++seed);   // ... order 1
    }

    struct Sw { int m, n, pad, k1, k2, incx; };
    const Sw sw[] = {
        {40, 33, 3, 1, 40, 1},  {40, 33, 0, 5, 31, 1},  {40, 33, 4, 1, 40, 2},  {40, 33, 0, 3, 29, 2},
        {40, 33, 2, 1, 40, -1}, {40, 33, 0, 5, 31, -2}, {40, 1, 0, 1, 40, 1},   {7, 5, 1, 2, 2, 1},
        {64, 130, 0, 1, 64, 1}, {64, 130, 3, 2, 63, -1},
    };
    for (const Sw& c : sw) failures += check_laswp<Backend, T>(h, c.m, c.n, c.pad, c.k1, c.k2, c.incx, ++seed);

    failures += check_empty<Backend, T>(h);
    return failures;
}

template <DdlaBackend Backend>
int run_backend()
{
    DdlaHandle_t h = nullptr;
    ddlaInit(h, Backend);
    ddlaSet(h, MPI_COMM_WORLD);
    int failures = 0;
    failures += run_type<Backend, float>(h);
    failures += run_type<Backend, double>(h);
    failures += run_type<Backend, std::complex<float>>(h);
    failures += run_type<Backend, std::complex<double>>(h);
    ddlaDestroy(h);
    if (g_rank == 0)
        std::cout << "  " << (Backend == DdlaBackend::CPU ? "CPU" : "GPU") << " backend: "
                  << (failures == 0 ? "ok" : "FAILED") << std::endl;
    return failures;
}

// -- argument checks --------------------------------------------------------------------
template <DdlaBackend Backend>
int check_rejections(const DdlaHandle_t& h, const DdlaHandle_t& other)
{
    // Argument checks fire before any pointer is dereferenced, so host buffers are fine on either backend.
    double x[64] = {0};
    int ip[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    int info = 0;
    int failures = 0;
    auto expect_throw = [&](const std::function<void()>& fn, const char* what) {
        bool threw = false;
        try { fn(); }
        catch (const std::runtime_error& e) {
            threw = std::string(e.what()).find(what) != std::string::npos;
        }
        if (!threw) {
            if (g_rank == 0) std::cerr << "  FAIL rejection: expected \"" << what << "\"" << std::endl;
            ++failures;
        }
    };
    // getrf
    expect_throw([&] { getrf<Backend, double>(nullptr, 2, 2, x, 2, ip, &info); }, "null handle");
    expect_throw([&] { getrf<Backend, double>(h, -1, 2, x, 2, ip, &info); }, "negative");
    expect_throw([&] { getrf<Backend, double>(h, 2, 2, x, 2, ip, nullptr); }, "null info");
    expect_throw([&] { getrf<Backend, double>(h, 2, 2, nullptr, 2, ip, &info); }, "null A");
    expect_throw([&] { getrf<Backend, double>(h, 2, 2, x, 2, nullptr, &info); }, "null ipiv");
    expect_throw([&] { getrf<Backend, double>(h, 4, 2, x, 3, ip, &info); }, "lda too small");
    // potrf
    expect_throw([&] { potrf<Backend, double>(nullptr, 'L', 2, x, 2, &info); }, "null handle");
    expect_throw([&] { potrf<Backend, double>(h, 'X', 2, x, 2, &info); }, "invalid uplo");
    expect_throw([&] { potrf<Backend, double>(h, 'L', -1, x, 2, &info); }, "negative");
    expect_throw([&] { potrf<Backend, double>(h, 'L', 2, x, 2, nullptr); }, "null info");
    expect_throw([&] { potrf<Backend, double>(h, 'L', 2, nullptr, 2, &info); }, "null A");
    expect_throw([&] { potrf<Backend, double>(h, 'U', 4, x, 3, &info); }, "lda too small");
    // laswp
    expect_throw([&] { laswp<Backend, double>(nullptr, 2, x, 4, 1, 2, ip, 1); }, "null handle");
    expect_throw([&] { laswp<Backend, double>(h, 2, x, 4, 0, 2, ip, 1); }, "k1 must be");
    expect_throw([&] { laswp<Backend, double>(h, 2, x, 4, 1, 2, ip, 0); }, "incx must be nonzero");
    expect_throw([&] { laswp<Backend, double>(h, 2, nullptr, 4, 1, 2, ip, 1); }, "null A");
    expect_throw([&] { laswp<Backend, double>(h, 2, x, 4, 1, 2, nullptr, 1); }, "null ipiv");
    expect_throw([&] { laswp<Backend, double>(h, 2, x, 3, 1, 4, ip, 1); }, "lda too small");
    // swap
    expect_throw([&] { swap<Backend, double>(nullptr, 2, x, 1, x + 8, 1); }, "null handle");
    expect_throw([&] { swap<Backend, double>(h, 2, nullptr, 1, x + 8, 1); }, "null x or y");
    expect_throw([&] { swap<Backend, double>(h, 2, x, 0, x + 8, 1); }, "nonzero");
    expect_throw([&] { swap<Backend, double>(h, 2, x, 1, x + 8, 0); }, "nonzero");
    if (other != nullptr) {
        expect_throw([&] { getrf<Backend, double>(other, 2, 2, x, 2, ip, &info); }, "does not match");
        expect_throw([&] { potrf<Backend, double>(other, 'L', 2, x, 2, &info); }, "does not match");
        expect_throw([&] { laswp<Backend, double>(other, 2, x, 4, 1, 2, ip, 1); }, "does not match");
        expect_throw([&] { swap<Backend, double>(other, 2, x, 1, x + 8, 1); }, "does not match");
    }
    return failures;
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);

    const char* only = std::getenv("DDLA_TEST_ONLY");
    const bool do_cpu = (!only || std::string(only) == "cpu");
    const bool do_gpu = (!only || std::string(only) == "gpu");
    int failures = 0;
#if DDLA_HAS_CPU
    if (do_cpu) failures += run_backend<DdlaBackend::CPU>();
#else
    (void)do_cpu;
#endif
#if DDLA_HAS_GPU
    if (do_gpu) failures += run_backend<DdlaBackend::GPU>();
#else
    (void)do_gpu;
#endif

    {   // argument checks: one handle of each available backend; in a dual build each is also handed the
        // other backend's handle to prove the mismatch is rejected.
        DdlaHandle_t cpu = nullptr, gpu = nullptr;
#if DDLA_HAS_CPU
        ddlaInit(cpu, DdlaBackend::CPU);
        ddlaSet(cpu, MPI_COMM_WORLD);
#endif
#if DDLA_HAS_GPU
        ddlaInit(gpu, DdlaBackend::GPU);
        ddlaSet(gpu, MPI_COMM_WORLD);
#endif
#if DDLA_HAS_CPU
        failures += check_rejections<DdlaBackend::CPU>(cpu, gpu);
#endif
#if DDLA_HAS_GPU
        failures += check_rejections<DdlaBackend::GPU>(gpu, cpu);
#endif
        if (cpu != nullptr) ddlaDestroy(cpu);
        if (gpu != nullptr) ddlaDestroy(gpu);
    }

    int global_failures = 0;
    MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (g_rank == 0) {
        if (global_failures == 0)
            std::cout << "test_backend_lapack passed" << std::endl;
        else
            std::cout << "test_backend_lapack FAILED: " << global_failures << " case(s)" << std::endl;
    }
    MPI_Finalize();
    return global_failures == 0 ? 0 : 1;
}
