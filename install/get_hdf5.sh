#!/bin/bash

# when changing dependency versions,
# make sure that the related paths in the Makefile are still valid.
hdf5_ver="184445f"

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

get_hdf5() {
	local ver="$1"
	# download hdf5 release and extract, if it doesn't exist
	local pname="hdf5"
	if [ -d "$pname" ]; then
		echo "Existing directory \"$pname\" found. Skipping download and configuration..."
	else
		echo "Downloading $pname..."
		wd="$(pwd -P)"
		local archive_base="hdf5-develop-${ver}"
		local archive="${archive_base}.tar.gz"
		echo "-----------------------"
		echo $archive
		echo "-----------------------"
		wget "https://github.com/HDFGroup/hdf5/releases/download/snapshot/$archive" 
		tar -xvzf "$archive"
		mv "hdfsrc" "hdf5"

		# rename, configure and build
		mv "$archive_base" $pname
		cd $pname
		./configure

		# get number of threads to build with
		np=$(get_n_threads_build)
		make -j $np

		make install 


		# # remove archive
		rm "$wd/$archive"
	fi
}

get_hdf5 "$hdf5_ver"
