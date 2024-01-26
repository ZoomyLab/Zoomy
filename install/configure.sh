#!/bin/bash

set -eu

# Default paths for dependencies.
# Will be written to Makefile, therefore variable expansion is suppressed.
# These are overwritten if the respective arguments are passed to the script.


# Create the dependencies directory if it does not exist
if [ ! -d "$../dependencies" ]; then
	mkdir -p "../dependencies"
fi

kokkos_path_def='${VOLKOSPATH}/dependencies/kokkos'
eigen_path_def='${VOLKOSPATH}/dependencies/eigen'
pnetcdf_path_def='${VOLKOSPATH}/dependencies/pnetcdf/src'
hdf5_path_def='${VOLKOSPATH}/dependencies/hdf5/hdf5'

# variable names in the Makefile
kokkos_path_var="KOKKOS_PATH"
eigen_path_var="EIGEN_PATH"
pnetcdf_path_var="PNETCDF_PATH"
hdf5_path_var="HDF5_PATH"

# flag for default options only
alldefault=0
# declare empty, because zero-length is used as condition later
kokkos_path=
eigen_path=
pnetcdf_path=
hdf5_path=

# use getopt to parse short and long options
! getopt --test > /dev/null
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
	echo "Enhanced getopt not available, defaulting to all-default option."
	alldefault=1
else
	VALID_ARGS=$(getopt \
	-o hak:p: \
	--long help,all-default,kokkos:,eigen:, hdf5:,  \
	-- "$@")
	if [[ $? -ne 0 ]]; then
		echo "getopt failed to parse arguments. Exiting..."
		exit 1;
	fi

	eval set -- "$VALID_ARGS"
	while [ : ]; do
		case "$1" in
			-h | --help)
echo -n "This script is intended to help configuring the VOLKOS compilation.
Valid options are:
  -a, --all-default  : Use default paths. Downloads and configures
                       dependencies as sub-paths of VOLKOS.
  -h, --help         : Show options
  -k, --kokkos       : Provide path to existing kokkos installation.
                       Default: \"$kokkos_path_def\"
  -e, --eigen      : Provide path to existing eigen installation.
                       Default: \"$eigen_path_def\"
  -h, --hdf5      : Provide path to existing hdf5 installation.
                       Default: \"$hdf5_path_def\"
  -p, --pnetcdf      : Provide path to existing pnetcdf installation.
                       Default: \"$pnetcdf_path_def\"

If a relative path is provided, it is interpreted as relative to
VOLKOS's root directory.
Paths can also be explicitly defaulted, by using option
argument \"default\".
If no arguments are provided, the script enters an interactive mode
for specifying the paths.
"
				exit
				;;
			-a | --all-default)
				echo "Using default paths..."
				alldefault=1
				break
				;;
			-k | --kokkos)
				if [ "$2" = "default" ]; then
					kokkos_path="$kokkos_path_def"
				else
					kokkos_path="$2"
				fi
				echo "kokkos path: \"$kokkos_path\"."
				shift 2
				;;
			-e | --eigen)
				if [ "$2" = "default" ]; then
					eigen_path="$eigen_path_def"
				else
					eigen_path="$2"
				fi
				echo "eigen path argument: \"$eigen_path\"."
				shift 2
				;;
			--) shift;
				break
				;;
			-p | --pnetcf)
				if [ "$2" = "default" ]; then
					pnetcdf_path="$pnetcdf_path_def"
				else
					pnetcdf_path="$2"
				fi
				echo "pnetcdf path argument: \"$pnetcdf_path\"."
				shift 2
				;;
			-h | --hdf5)
				if [ "$2" = "default" ]; then
					hdf5_path="$hdf5_path_def"
				else
					hdf5_path="$2"
				fi
				echo "hdf5 path argument: \"$hdf5_path\"."
				shift 2
				;;
			--) shift;
				break
				;;
		esac
	done
fi

# script dir, to get consistent execution, even from another location
sd="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

sf="$sd/funcs.sh"
if [ -f "$sf" ]; then
	source "$sf"
else
	echo "Script file not found: $sf"
	echo "Exiting..."
	exit
fi

# check requirements (programs/modules)
echo -e "\nChecking prerequisites..."
require git wget g++ mpicc make m4
# if a minimum version is required, use this:
# require_version g++ 9.3 mpicc 9.3
require_env_var MPICC
echo -e "Prerequisites met.\n"


if ((alldefault == 0)) &&  [[ ( -z "$kokkos_path" || -z "$eigen_path" || -z "$pnetcdf_path" || -z "$hdf5_path") ]]; then
	echo -n \
"Entering interactive mode...
-------------------------------------------------
You will be asked to specify one or multiple paths for compilation dependencies.
If a dependency is not yet installed, choose the default option.
This will download and install the dependency into a subfolder of VOLKOS.
If a dependency is already installed, choose to provide a custom path.
All paths may also instead be provided as command line options.
Use the --help option for more information.

The dependency paths are entered into the Makefile by this script.
-------------------------------------------------
"
fi

# go to volkos dir
cd "$sd/.."
path_varname="VOLKOSPATH"
export VOLKOSPATH="$(pwd -P)"

check_path() {
	local path_inp="$1"
	local path=
	# must be either absolute, or relative to VOLKOSPATH
	if [[ ( "${path_inp:0:1}" = "/" && -d "$path_inp" ) \
	|| "$path_inp" = '${VOLKOSPATH}'* ]]
	then
		# if ${VOLKOSPATH} is part of the path,
		# then no check is performed, because the directory
		# may not exist yet.
		path="$path_inp"
	elif [ -d "$VOLKOSPATH/$path_inp" ]; then
		path='${VOLKOSPATH}'"/$path_inp"
	else
		echo "Directory \"$path_inp\" not found!" >&2
		path=
	fi
	echo $path
}

ask_or_default() {
	local path="$1"
	local path_def="$2"
	local path_name="$3"
	local path_inp
	# default path, no check required
	if ((alldefault == 1)); then
		path="$path_def"
	# interactive path specification
	elif [ -z "$path" ]; then
		printf -- "Please specify the value for %s to be used,\n%s\n" \
		$path_name "by entering the corresponding option number." >&2
		choice=$(get_user_choice_def true "$path_def" \
		"Custom path (a prompt will appear)")
		if ((choice == 0)); then
			path="$path_def"
		else
			while :; do
				printf -- "Please enter the value for %s,\n%s%s:\n" \
				$path_name \
				"either as an absolute path or relative to " \
				$VOLKOSPATH >&2
				read path_inp
				path=$(check_path "$path_inp")
				if [ -n "$path" ]; then
					break
				fi
			done
		fi
		echo "" >&2
	# path as command line argument
	else
		path=$(check_path "$path")
		if [ -z $path ]; then
			echo "Exiting script..." >&2
			exit 1
		fi
	fi
	echo "$path"
}

kokkos_path=$(ask_or_default "$kokkos_path" "$kokkos_path_def" "$kokkos_path_var")
eigen_path=$(ask_or_default "$eigen_path" "$eigen_path_def" "$eigen_path_var")
pnetcdf_path=$(ask_or_default "$pnetcdf_path" "$pnetcdf_path_def" "$pnetcdf_path_var")
hdf5_path=$(ask_or_default "$hdf5_path" "$hdf5_path_def" "$hdf5_path_var")

echo "Paths: "
echo "$path_varname=\"$VOLKOSPATH\""
echo "$kokkos_path_var=\"$kokkos_path\""
echo "$eigen_path_var=\"$eigen_path\""
echo "$pnetcdf_path_var=\"$pnetcdf_path\""
echo "$hdf5_path_var=\"$hdf5_path\""


makefile="$VOLKOSPATH/library/solver/Makefile"

replace_path() {
	local path_var="$1"
	local path="$2"
	sed -i "s|$path_var\s\?=\s\?.*|$path_var=$path|" "$makefile"
}

if [ -f "$makefile" ]; then
	echo "Entering paths into Makefile..."
	replace_path "$kokkos_path_var" "$kokkos_path"
	replace_path "$eigen_path_var" "$eigen_path"
	replace_path "$pnetcdf_path_var" "$pnetcdf_path"
	replace_path "$hdf5_path_var" "$hdf5_path"
else
	echo "Makefile not found at: $makefile"
	echo "Exiting..."
	exit
fi


echo ""

# download dependencies, if required

if [ "$kokkos_path" = "$kokkos_path_def" ]; then
# 	get_kokkos "$kokkos_ver"
  cd ${VOLKOSPATH}/dependencies
	"$sd/get_kokkos.sh"
	echo ""
  cd ${VOLKOSPATH}
fi

if [ "$eigen_path" = "$eigen_path_def" ]; then
  cd ${VOLKOSPATH}/dependencies
	"$sd/get_eigen.sh"
	echo ""
  cd ${VOLKOSPATH}
fi

if [ "$hdf5_path" = "$hdf5_path_def" ]; then
  cd ${VOLKOSPATH}/dependencies
	"$sd/get_hdf5.sh"
	echo ""
  cd ${VOLKOSPATH}
fi

if [ "$pnetcdf_path" = "$pnetcdf_path_def" ]; then
  cd ${VOLKOSPATH}/dependencies
	"$sd/get_pnetcdf.sh"
	echo ""
  cd ${VOLKOSPATH}
fi


# put VOLKOSPATH in run control file

# for that, get default shell
defsh="$(basename -- "${SHELL}")"
echo "Found default shell: ${defsh}"

rcfile="$HOME/.${defsh}rc"
# echo "${rcfile}"

if [ ! -f "${rcfile}" ]; then
	echo "No run control file found!"
	echo "Cannot register ${path_varname}."
	exit
fi

echo "Registering ${path_varname}..."
echo "Using run control file: ${rcfile}"

# create a comment in the file to make it easier to identify later
expcomment="# this variable was added automatically by VOLKOS"
expstem="export ${path_varname}="

# check whether export command already exist, including the comment.
grep_export() {
	grep $1 "${expstem}\|${expcomment}" "${rcfile}"
}

if grep_export -q; then
	echo "Found previous registration of ${path_varname}. Overwriting..."
	sed -i "/${expcomment}/d" "${rcfile}"
	sed -i "/${expstem}/d" "${rcfile}"
fi

echo "${expcomment}" >> "${rcfile}"
echo "${expstem}${VOLKOSPATH}" >> "${rcfile}"

echo "
-------------------------------------------------
Finished configuring VOLKOS. To compile VOLKOS,
first source your run control file, e.g.
   source "${rcfile}"
then navigate to:
   ${VOLKOSPATH}/library/solver
and run make, e.g.
   make
Consult the Makefile for compilation options.
-------------------------------------------------"

