// misc.h
#ifndef MISC_H
#define MISC_H

#include "define.h"
#include <iostream>
#include <iomanip>
#include <algorithm>

void print_string(std::string str)
{
    std::cout << str << std::endl;
}

template <typename Arr2>
void print_matrix(const Arr2 &matrix, int max_i = 999, int max_j = 999)
{

    int i_max = std::min((int)matrix.extent(0), max_i);
    int j_max = std::min((int)matrix.extent(1), max_j);
    std::cout << "\n\nMatrix size: (" << matrix.extent(0) << ", " << matrix.extent(1) << ")" << std::endl;
    for (int i = 0; i < i_max; ++i)
    {
        for (int j = 0; j < j_max; ++j)
        {
            std::cout << std::left << std::setw(10) << matrix(i, j) << "  ";
        }
    std::cout << std::endl;
    }
}

#endif // MISC_H