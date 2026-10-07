// EMITTED from the NSM: state/aux/parameter layout
#pragma once
constexpr int N_VARS = 3;
constexpr int N_AUX = 4;
constexpr int N_PARAMS = 5;
constexpr int IDX_b = 0;
constexpr int IDX_h = 1;
constexpr int IDX_q_0 = 2;
constexpr int AUX_dq0dx = 0;
constexpr int AUX_dhdx = 1;
constexpr int AUX_dbdx = 2;
constexpr int AUX_hinv = 3;
constexpr int PAR_g = 0;
constexpr int PAR_rho = 1;
constexpr int PAR_nu = 2;
constexpr int PAR_lambda_s = 3;
constexpr int PAR_e_x = 4;
constexpr double PARAM_VALUES[N_PARAMS] = {9.81, 1.0, 0.0, 0.0, 0.0};
