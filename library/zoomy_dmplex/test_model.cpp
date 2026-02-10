#include <iostream>
#include <vector>
#include <cmath>
#include "Model.H" 

// Mock implementation if simple array is missing in standalone compile
#ifndef ZOOMY_SIMPLE_ARRAY
// (Model.H should define this, but just in case)
#endif

int main() {
    using Real = double;
    
    // 1. Setup Dummy State
    Real h = 1.0;
    Real u = 1.0;
    Real v = 0.5;
    
    Real Q[6] = {0.0, h, h*u, h*v, 0.0, 0.0};
    Real Qaux[1] = {1.0/h}; // 1.0
    Real p[20] = {0}; // Dummy params
    
    // 2. Call the Matrix Function
    auto res = Model<Real>::nonconservative_matrix(Q, Qaux, p);
    
    // 3. Check specific indices
    // We expect Interleaved Layout: (Row, Col, Dim) -> r*12 + c*2 + d
    
    std::cout << "--- Integrity Check ---" << std::endl;
    
    // Check Mass Equation (Row 1) - Should be ALL Zero
    bool mass_ok = true;
    for(int c=0; c<6; ++c) {
        for(int d=0; d<2; ++d) {
            int idx = 1*12 + c*2 + d;
            if(std::abs(res[idx]) > 1e-12) {
                std::cout << "[FAIL] Mass Entry (Row 1, Col " << c << ", Dim " << d 
                          << ") at Index " << idx << " is " << res[idx] << std::endl;
                mass_ok = false;
            }
        }
    }
    if(mass_ok) std::cout << "[PASS] Mass Equation (Row 1) is purely zero." << std::endl;

    // Check Momentum X (Row 3, Col 3, Dim 0) - Should be -u
    int idx_mom = 3*12 + 3*2 + 0; // Index 42
    std::cout << "[INFO] Momentum Term at Index " << idx_mom << " is " << res[idx_mom] 
              << " (Expected -1.0)" << std::endl;

    // Check for "Blocked" Leakage
    // In Blocked layout, index 14 is (Row 2, Col 2, Dim 0) -> Momentum.
    // In Interleaved, index 14 is (Row 1, Col 1, Dim 0) -> Mass.
    std::cout << "[INFO] Value at Index 14 is " << res[14] << std::endl;
    
    if (std::abs(res[14]) > 1e-9) {
        std::cout << "[CRITICAL] Index 14 is non-zero! The code thinks this is Mass, but it contains data." << std::endl;
    } else {
        std::cout << "[PASS] Index 14 is zero." << std::endl;
    }

    return 0;
}