#!/bin/bash

# when changing dependency versions,
# make sure that the related paths in the Makefile are still valid.
kokkos_ver="3.7.01"

get_kokkos() {
	local ver="$1"
	# clone KOKKOS, if no kokkos dir exists
	if [ -d "kokkos" ]; then
		echo "Existing directory \"dependencies/kokkos\" found. Skipping download..."
	else
		echo "Cloning kokkos..."
		git clone --depth 1 --branch $ver https://github.com/kokkos/kokkos.git
	fi
}

get_kokkos "$kokkos_ver"
