#!/bin/bash

# when changing dependency versions,
# make sure that the related paths in the Makefile are still valid.
eigen_ver="3.4.0"

get_eigen() {
	local ver="$1"
	# download eigen release and extract, if it doesn't exist
	local pname="eigen"
	if [ -d "$pname" ]; then
		echo "Existing directory \"$pname\" found. Skipping download and configuration..."
	else
		echo "Downloading $pname..."
		wd="$(pwd -P)"
		local archive_base="eigen-${ver}"
		local archive="${archive_base}.tar.gz"
    wget "https://gitlab.com/libeigen/eigen/-/archive/${ver}/${archive}"
		tar -xvzf "$archive"
	fi
}

get_eigen "$eigen_ver"
