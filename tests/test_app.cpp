// This is the main test application for the C++ component

#include <iostream>

// Forward declarations of all test functions
void test_bernoulli();
void test_binomial();
void test_negative_binomial();
void test_beta();
void test_gamma();
void test_chi_square();
void test_discrete_uniform();
void test_exponential();
void test_geometric();
void test_normal();
void test_poisson();
void test_uniform();
void test_utils();

int main() {
    std::cout << "Starting fastdist C++ tests...\n";

    test_bernoulli();
    test_binomial();
    test_negative_binomial();
    test_discrete_uniform();
    test_exponential();
    test_geometric();
    test_normal();
    test_poisson();
    test_uniform();
    test_beta();
    test_gamma();
    test_chi_square();
    test_utils();

    std::cout << "All tests completed successfully.\n";
    return 0;
}
