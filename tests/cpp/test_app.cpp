// This is the main test application for the C++ component

#include <cstddef>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

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
void test_rng();

// Run one case by name, or all of them when given no argument. Keeping the
// list ordered (rather than a map) preserves the original execution order.
int main(int argc, char** argv) {
    const std::vector<std::pair<std::string, void (*)()>> tests{
            {"bernoulli", test_bernoulli},
            {"binomial", test_binomial},
            {"negative_binomial", test_negative_binomial},
            {"discrete_uniform", test_discrete_uniform},
            {"exponential", test_exponential},
            {"geometric", test_geometric},
            {"normal", test_normal},
            {"poisson", test_poisson},
            {"uniform", test_uniform},
            {"beta", test_beta},
            {"gamma", test_gamma},
            {"chi_square", test_chi_square},
            {"utils", test_utils},
            {"rng", test_rng},
    };

    const auto lookup = [&tests](const std::string& name) -> void (*)() {
        for (const auto& [n, fn]: tests) {
            if (n == name) {
                return fn;
            }
        }
        return nullptr;
    };

    // Any number of names may be given. Resolve them all before running
    // anything, so a typo in the last argument does not leave the earlier
    // cases half-run.
    if (argc > 1) {
        std::vector<void (*)()> selected;
        selected.reserve(static_cast<std::size_t>(argc - 1));

        for (int i = 1; i < argc; ++i) {
            void (*fn)() = lookup(argv[i]);
            if (fn == nullptr) {
                std::cerr << "unknown test: " << argv[i] << '\n';
                return 2;
            }
            selected.push_back(fn);
        }

        for (const auto fn: selected) {
            fn();
        }
        return 0;
    }

    std::cout << "Starting fastdist C++ tests...\n";

    for (const auto& [name, fn]: tests) {
        fn();
    }

    std::cout << "All tests completed successfully.\n";
    return 0;
}
