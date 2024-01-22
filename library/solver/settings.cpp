// Settings.cpp
#include "Settings.h"
#include <H5Cpp.h>

Settings::Settings(const std::string &filename)
{
    // Open the HDF5 file
    H5::H5File file(filename, H5F_ACC_RDONLY);

    // Read the 'name' attribute
    H5::Attribute attr = file.openAttribute("name");
    H5::StrType strtype = attr.getStrType();
    attr.read(strtype, this->name);

    // Read the 'parameters' dataset
    H5::DataSet dataset = file.openDataSet("parameters");
    H5::DataSpace dataspace = dataset.getSpace();
    hsize_t length = dataspace.getSimpleExtentNpoints();
    this->parameters.resize(length);
    dataset.read(this->parameters.data(), H5::PredType::NATIVE_FLOAT);
}