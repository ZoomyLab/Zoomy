// test.cpp
extern "C" void myfunc(const double* q, double* out) {
    out[0] = q[0] * q[1];
}

