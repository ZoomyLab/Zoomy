#!/bin/bash

require() {
	reqMissing=0
	for prog in "$@"
	do
		echo -n "  Checking $prog ... "
		if ! command -v "$prog" &> /dev/null
		then
			echo "NOT FOUND! Exiting after requirement check is finished."
			reqMissing=1
		else
			echo "OK"
		fi
	done
	if [ $reqMissing -eq 1 ]; then
		exit
	fi
}

require_env_var() {
	reqMissing=0
	for var in "$@"
	do
		echo -n "  Checking environment variable \"${var}\" ... "
		if [ -z "${var}" ]; then
			echo "NOT FOUND! Exiting after requirement check is finished."
			reqMissing=1
		else
			echo "OK"
		fi
	done
	if [ $reqMissing -eq 1 ]; then
		exit
	fi
}

get_version() {
	echo "$($1 --version | grep '([0-9]+\.)+[0-9]+' -Eo | head -1)"
}

version_is_greater() {
	local target="$1"
	local actual="$2"
	if [ "$(printf '%s\n' "$target" "$actual" | sort -V | head -n1)" == "$target" ]
	then
		echo 1 # I'd prefer a boolean true
	else
		echo 0
	fi
}

require_version() {
	if (( $# % 2 != 0 )); then
		echo "require_version: Number of arguments must be even!"
		exit
	fi
	local msg="Exiting after requirement check is finished."
	local reqMissing=0
	while true; do
		local prog="${1:-""}"
		local ver="${2:-""}"
		if [ -z "$prog" ] | [ -z "$ver" ]; then
			break;
		fi
		echo -n "  Checking $prog, min. v.$ver ... "
		if ! command -v "$prog" &> /dev/null
		then
			echo "ERROR: NOT FOUND!"
			echo "$msg"
			reqMissing=1
		else
			local ver_actual="$(get_version $prog)"
			if [[ "$(version_is_greater "$ver" "$ver_actual")" == 1 ]]; then
				echo "OK, found v.$ver_actual"
			else
				echo "ERROR: Only found v.$ver_actual!"
				echo "$msg"
				reqMissing=1
			fi
		fi
		shift 2
	done
	if [ $reqMissing -eq 1 ]; then
		exit
	fi
}

# Custom 'select' implementation that allows *empty* input.
# First argument should be true or false,
# for whether to output the option index or the option string,
# with "true" for the index and "false" for the string.
# Pass the options as individual arguments after that.
# Example:
#    choice=$(get_user_choice_def false 'one' 'two' 'three')
get_user_choice_def() {
    returnIndex=${1}
    shift 1

    local i numItems=$#

    # mark default option with this string
    local defstr="[default] "

    # Print numbered menu items, based on the arguments passed.
    for (( i=1; i<=$#; ++i )); do
        printf '%i) %s%s\n' $i "${defstr}" "${@:$i:1}"
        defstr=""
    done >&2 # Print to stderr, as `select` does.

    # Prompt the user for the index of the desired item.
    while :; do
        printf %s "${PS3-#? }" >&2 # Print the prompt string to stderr, as `select` does.
        read -r index
        # Make sure that the input is either empty or that a valid index was entered.
        [[ -z $index ]] && break  # empty input
        (( index >= 1 && index <= numItems )) 2>/dev/null || \
        { echo "${index} is not a valid choice. Please the number of one of the options above." >&2; continue; }
        break
    done
    # use first option as default
    if [[ -z $index ]]; then index=1; fi
    # and print it
    printf -- " -> Using option %i) %s\n" $index "${@:$index:1}" >&2

    # Output the selected item, either as index or as option string
    if [ $returnIndex = true ]; then
        echo $(( $index - 1))
    else
        printf %s "${@:$index:1}"
    fi
}

