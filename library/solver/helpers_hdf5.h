// helpers_hdf5.h
#ifndef HELPERS_HDF5_H
#define HELPERS_HDF5_H

#include "hdf5.h"
#include <string>
#include <vector>
#include <iostream>


hid_t openHdf5(const std::string& filePath) {
    hid_t file = H5Fopen(filePath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        std::cerr << "Error opening file: " << filePath << std::endl;
    } 
    return file;
}

void readDoubleFromDataset(hid_t &file, const std::string& datasetName, double& outputVar) {
    hid_t dataset = H5Dopen(file, datasetName.c_str(), H5P_DEFAULT);
    if (dataset < 0) 
    {
        std::cerr << "Error opening dataset: " << datasetName << std::endl;
    } 
    else 
    {
        H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &outputVar);
        H5Dclose(dataset);
    }
}

void readIntFromDataset(hid_t &file, const std::string& datasetName, int& outputVar) {
    hid_t dataset = H5Dopen(file, datasetName.c_str(), H5P_DEFAULT);
    if (dataset < 0) 
    {
        std::cerr << "Error opening dataset: " << datasetName << std::endl;
    } 
    else 
    {
        H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &outputVar);
        H5Dclose(dataset);
    }
}


void readBoolFromDataset(hid_t &file, const std::string& datasetName, bool& outputVar) {
    hid_t dataset = H5Dopen(file, datasetName.c_str(), H5P_DEFAULT);
    if (dataset < 0) 
    {
        std::cerr << "Error opening dataset: " << datasetName << std::endl;
    } 
    else 
    {
        H5Dread(dataset, H5T_NATIVE_HBOOL, H5S_ALL, H5S_ALL, H5P_DEFAULT, &outputVar);
        H5Dclose(dataset);
    }
}

void readDoubleArrayFromDataset(hid_t file, const std::string& datasetName, std::vector<double>& outputArray) {
    hid_t dataset = H5Dopen(file, datasetName.c_str(), H5P_DEFAULT);
    if (dataset < 0) 
    {
        std::cerr << "Error opening dataset: " << datasetName << std::endl;
    } 
    else 
    {
        // Get the size of the dataset
        hid_t dataspace = H5Dget_space(dataset);
        hsize_t size = H5Sget_simple_extent_npoints(dataspace);

        // Resize the output array to match the size of the dataset
        outputArray.resize(size);

        // Read the dataset
        H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, outputArray.data());

        // Close the dataspace and dataset
        H5Sclose(dataspace);
        H5Dclose(dataset);
    }
}

void readIntArrayFromDataset(hid_t file, const std::string& datasetName, std::vector<int>& outputArray) {
    hid_t dataset = H5Dopen(file, datasetName.c_str(), H5P_DEFAULT);
    if (dataset < 0) 
    {
        std::cerr << "Error opening dataset: " << datasetName << std::endl;
    } 
    else 
    {
        // Get the size of the dataset
        hid_t dataspace = H5Dget_space(dataset);
        hsize_t size = H5Sget_simple_extent_npoints(dataspace);

        // Resize the output array to match the size of the dataset
        outputArray.resize(size);

        // Read the dataset
        H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, outputArray.data());

        // Close the dataspace and dataset
        H5Sclose(dataspace);
        H5Dclose(dataset);
    }
}
    

void readStringFromDataset(hid_t file, const std::string& datasetName, std::string& outputString) {
    hid_t dataset = H5Dopen(file, datasetName.c_str(), H5P_DEFAULT);
    if (dataset < 0) 
    {
        std::cerr << "Error opening dataset: " << datasetName << std::endl;
    } 
    else 
    {
        // Get the native datatype
        hid_t filetype = H5Dget_type(dataset);
        hid_t memtype = H5Tget_native_type(filetype, H5T_DIR_ASCEND);

        // Get the dataspace
        hid_t dataspace = H5Dget_space(dataset);

        // Read the dataset
        char* buffer;
        H5Dread(dataset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &buffer);

        // Assign the buffer to the output string
        outputString = buffer;

        // Free the buffer
        H5Dvlen_reclaim(memtype, dataspace, H5P_DEFAULT, &buffer);

        // Close the datatype and dataset
        H5Tclose(memtype);
        H5Tclose(filetype);
        H5Sclose(dataspace);
        H5Dclose(dataset);
    }
}


void readDouble2dArrayFromDataset(hid_t file, const std::string& datasetName, std::vector<std::vector<double>>& outputArray) {
    hid_t dataset = H5Dopen(file, datasetName.c_str(), H5P_DEFAULT);
    if (dataset < 0) 
    {
        std::cerr << "Error opening dataset: " << datasetName << std::endl;
    } 
    else 
    {
        // Get the dimensions of the dataset
        hid_t dataspace = H5Dget_space(dataset);
        int ndims = H5Sget_simple_extent_ndims(dataspace);
        hsize_t dims[ndims];
        H5Sget_simple_extent_dims(dataspace, dims, NULL);

        // Resize the output array to match the dimensions of the dataset
        outputArray.resize(dims[0], std::vector<double>(dims[1]));

        // Create a temporary buffer to hold the data
        double* buffer = new double[dims[0] * dims[1]];

        // Read the dataset
        H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);

        // Copy the data from the buffer into the output array
        for (hsize_t i = 0; i < dims[0]; ++i)
            for (hsize_t j = 0; j < dims[1]; ++j)
                outputArray[i][j] = buffer[i * dims[1] + j];

        // Delete the buffer
        delete[] buffer;

        // Close the dataspace and dataset
        H5Sclose(dataspace);
        H5Dclose(dataset);
    }
}

void readInt2dArrayFromDataset(hid_t file, const std::string& datasetName, std::vector<std::vector<int>>& outputArray) {
    hid_t dataset = H5Dopen(file, datasetName.c_str(), H5P_DEFAULT);
    if (dataset < 0) 
    {
        std::cerr << "Error opening dataset: " << datasetName << std::endl;
    } 
    else 
    {
        // Get the dimensions of the dataset
        hid_t dataspace = H5Dget_space(dataset);
        int ndims = H5Sget_simple_extent_ndims(dataspace);
        hsize_t dims[ndims];
        H5Sget_simple_extent_dims(dataspace, dims, NULL);

        // Resize the output array to match the dimensions of the dataset
        outputArray.resize(dims[0], std::vector<int>(dims[1]));

        // Create a temporary buffer to hold the data
        int* buffer = new int[dims[0] * dims[1]];

        // Read the dataset
        H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);

        // Copy the data from the buffer into the output array
        for (hsize_t i = 0; i < dims[0]; ++i)
            for (hsize_t j = 0; j < dims[1]; ++j)
                outputArray[i][j] = buffer[i * dims[1] + j];

        // Delete the buffer
        delete[] buffer;

        // Close the dataspace and dataset
        H5Sclose(dataspace);
        H5Dclose(dataset);
    }
}

void readDouble3dArrayFromDataset(hid_t file, const std::string& datasetName, std::vector<std::vector<std::vector<double>>>& outputArray) {
    hid_t dataset = H5Dopen(file, datasetName.c_str(), H5P_DEFAULT);
    if (dataset < 0) 
    {
        std::cerr << "Error opening dataset: " << datasetName << std::endl;
    } 
    else 
    {
        // Get the dimensions of the dataset
        hid_t dataspace = H5Dget_space(dataset);
        int ndims = H5Sget_simple_extent_ndims(dataspace);
        hsize_t dims[ndims];
        H5Sget_simple_extent_dims(dataspace, dims, NULL);

        // Resize the output array to match the dimensions of the dataset
        outputArray.resize(dims[0], std::vector<std::vector<double>>(dims[1], std::vector<double>(dims[2])));

        // Create a temporary buffer to hold the data
        double* buffer = new double[dims[0] * dims[1] * dims[2]];

        // Read the dataset
        H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);

        // Copy the data from the buffer into the output array
        for (hsize_t i = 0; i < dims[0]; ++i)
            for (hsize_t j = 0; j < dims[1]; ++j)
                for (hsize_t k = 0; k < dims[2]; ++k)
                    outputArray[i][j][k] = buffer[i * dims[1] * dims[2] + j * dims[2] + k];

        // Delete the buffer
        delete[] buffer;

        // Close the dataspace and dataset
        H5Sclose(dataspace);
        H5Dclose(dataset);
    }
}

void readStringArrayFromDataset(hid_t file, const std::string& datasetName, std::vector<std::string>& outputArray) {
    hid_t dataset = H5Dopen(file, datasetName.c_str(), H5P_DEFAULT);
    if (dataset < 0) 
    {
        std::cerr << "Error opening dataset: " << datasetName << std::endl;
    } 
    else 
    {
        // Get the native datatype
        hid_t filetype = H5Dget_type(dataset);
        hid_t memtype = H5Tget_native_type(filetype, H5T_DIR_ASCEND);

        // Get the dimensions of the dataset
        hid_t dataspace = H5Dget_space(dataset);
        hsize_t dims[1];
        H5Sget_simple_extent_dims(dataspace, dims, NULL);

        // Resize the output vector to match the dimensions of the dataset
        outputArray.resize(dims[0]);



        // Check if the datatype is variable-length or fixed-length
        if (H5Tis_variable_str(memtype)) 
        {
            // Handle variable-length strings

            // Create a temporary buffer to hold the data
            char** buffer = new char*[dims[0]];

            // Read the dataset into the buffer
            H5Dread(dataset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);

            // Copy the data from the buffer into the output vector
            for (hsize_t i = 0; i < dims[0]; ++i)
                outputArray[i] = buffer[i];

            // Reclaim the memory for the variable-length data
            H5Dvlen_reclaim(memtype, dataspace, H5P_DEFAULT, buffer);

            // Delete the buffer
            delete[] buffer;
        }
        else 
        {
            // Handle fixed-length strings
            size_t size = H5Tget_size(memtype);
            char* buffer = new char[dims[0] * size];

            // Read the dataset into the buffer
            H5Dread(dataset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);

            // Copy the data from the buffer into the output array
            for (hsize_t i = 0; i < dims[0]; ++i) {
                std::string str(&buffer[i * size], size);
                outputArray[i] = str.c_str();  // Use c_str() to get a null-terminated string
            }
            // Delete the buffer
            delete[] buffer;

        }

        // Close the datatype, dataspace and dataset
        H5Tclose(memtype);
        H5Tclose(filetype);
        H5Sclose(dataspace);
        H5Dclose(dataset);
    }
}


double loadFieldFromHdf5(hid_t& file, int index, std::vector<std::vector<double>>& Q, std::vector<std::vector<double>>& Qaux)
{
    std::string groupName = std::to_string(index);
    hid_t group = H5Gopen(file, groupName.c_str(), H5P_DEFAULT);
    if (group < 0) 
    {
        std::cerr << "Error opening group: " << groupName << std::endl;
    } 
    else 
    {
        readDouble2dArrayFromDataset(group, "Q", Q);
        readDouble2dArrayFromDataset(group, "Qaux", Qaux);
    }
    return 0.;
}


#endif // HELPERS_HDF5_H