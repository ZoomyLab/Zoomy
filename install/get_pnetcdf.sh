#!/bin/bash

# when changing dependency versions,
# make sure that the related paths in the Makefile are still valid.
pnetcdf_ver="1.12.3"

get_n_threads_build() {
	if ! command -v "lscpu" &> /dev/null
	then
		echo "Note: Command \"lscpu\" not found (MSYS shell?)." >&2
		echo "Defaulting to building with a single thread." >&2
		maxThreads=1
	else
		maxThreads=$(lscpu -p | grep -c "^[0-9]")
	fi
	echo $(( maxThreads - 2 >= 1 ? maxThreads - 2 : 1 ))
}

get_pnetcdf() {
	local ver="$1"
	# download PnetCDF release and extract, if it doesn't exist
	local pname="PnetCDF"
	if [ -d "$pname" ]; then
		echo "Existing directory \"$pname\" found. Skipping download and configuration..."
	else
		echo "Downloading $pname..."
		wd="$(pwd -P)"
		local archive_base="pnetcdf-${ver}"
		local archive="${archive_base}.tar.gz"
		wget "https://parallel-netcdf.github.io/Release/${archive}"
		tar -xvzf "$archive"

		# rename, configure and build
		mv "$archive_base" $pname
		cd $pname
		./configure

		# get number of threads to build with
		np=$(get_n_threads_build)
		make -j $np

		# remove archive
		rm "$wd/$archive"
	fi
}

get_pnetcdf "$pnetcdf_ver"
