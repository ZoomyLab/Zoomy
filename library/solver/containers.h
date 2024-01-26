#ifndef CONTAINERS_H
#define CONTAINERS_H

template <template <typename...> class Container, typename Type>
class Matrix : public Container<Type> {
public:
    Matrix(std::size_t N, std::size_t M) : rows_(N), cols_(M), Container<Type>(N * M) {}

    Matrix(std::size_t N, std::size_t M, const Type& value) : rows_(N), cols_(M), Container<Type>(N * M, value) {}

    Type& operator()(std::size_t i, std::size_t j) {
        return this->at(i * cols_ + j);
    }

    const Type& operator()(std::size_t i, std::size_t j) const {
        return this->at(i * cols_ + j);
    }

    std::size_t getRows() const {
        return rows_;
    }

    std::size_t getCols() const {
        return cols_;
    }

private:
    std::size_t rows_, cols_;
};

template <template <typename, typename> class Container, typename Type>
class Matrix {
public:
    Matrix(std::size_t N, std::size_t M) : rows_(N), cols_(M), data_(N * M) {}

    Matrix(std::size_t N, std::size_t M, double value) : rows_(N), cols_(M), data_(N * M, value) {}

    double& operator()(std::size_t i, std::size_t j) {
        return data_[i * cols_ + j];
    }

    const double& operator()(std::size_t i, std::size_t j) const {
        return data_[i * cols_ + j];
    }

    std::size_t getRows() const {
        return rows_;
    }

    std::size_t getCols() const {
        return cols_;
    }

private:
    std::size_t rows_, cols_;
    std::vector<double> data_;
};

template <template <typename, typename> class Container, typename Type>
class Matrix : public Container<Container<Type, std::allocator<Type>>, std::allocator<Container<Type, std::allocator<Type>>>>
{
public:
    // Add additional functionality here
        Matrix(std::size_t N, std::size_t M) : Container<Container<Type, std::allocator<Type>>, std::allocator<Container<Type, std::allocator<Type>>>>(N, Container<Type, std::allocator<Type>>(M)) {}
        Matrix(std::size_t N, std::size_t M, Type value) : Container<Container<Type, std::allocator<Type>>, std::allocator<Container<Type, std::allocator<Type>>>>(N, Container<Type, std::allocator<Type>>(M, value)) {}
};

template <template <typename, typename> class Container, typename Type>
class Matrix3 : public Container<Container<Container<Type, std::allocator<Type>>, std::allocator<Container<Type, std::allocator<Type>>>>, std::allocator<Container<Container<Type, std::allocator<Type>>, std::allocator<Container<Type, std::allocator<Type>>>>>>
{
public:
    Matrix3(std::size_t N, std::size_t M, std::size_t K) : Container<Container<Container<Type, std::allocator<Type>>, std::allocator<Container<Type, std::allocator<Type>>>>, std::allocator<Container<Container<Type, std::allocator<Type>>, std::allocator<Container<Type, std::allocator<Type>>>>>>(N, Container<Container<Type, std::allocator<Type>>, std::allocator<Container<Type, std::allocator<Type>>>>(M, Container<Type, std::allocator<Type>>(K))) {}
    Matrix3(std::size_t N, std::size_t M, std::size_t K, Type value) : Container<Container<Container<Type, std::allocator<Type>>, std::allocator<Container<Type, std::allocator<Type>>>>, std::allocator<Container<Container<Type, std::allocator<Type>>, std::allocator<Container<Type, std::allocator<Type>>>>>>(N, Container<Container<Type, std::allocator<Type>>, std::allocator<Container<Type, std::allocator<Type>>>>(M, Container<Type, std::allocator<Type>>(K, value))) {}
};

typedef std::vector<double> VEC;
typedef std::vector<int> IVEC;

typedef Matrix<std::vector, double> MAT;
typedef Matrix<std::vector, int> IMAT;

typedef Matrix3<std::vector, double> MAT3;
typedef Matrix3<std::vector, int> IMAT3;



#endif // CONTAINERS_HH