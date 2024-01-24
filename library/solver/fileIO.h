// fileIO.h
#ifndef FILE_IO_H
#define FILE_IO_H

#include "hdf5.h"
#include <string>
#include <vector>
#include <iostream>

hid_t loadHDF5(const std::string& filePath) {
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


void readDouble2DArrayFromDataset(hid_t file, const std::string& datasetName, std::vector<std::vector<double>>& outputArray) {
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

        // Read the dataset
        H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, outputArray[0].data());

        // Close the dataspace and dataset
        H5Sclose(dataspace);
        H5Dclose(dataset);
    }
}


void readInt2DArrayFromDataset(hid_t file, const std::string& datasetName, std::vector<std::vector<int>>& outputArray) {
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

        // Read the dataset
        H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, outputArray[0].data());

        // Close the dataspace and dataset
        H5Sclose(dataspace);
        H5Dclose(dataset);
    }
}
#endif // FILE_IO_H