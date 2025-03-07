#!/bin/bash

Help() {
  echo "get_all_errors.sh -r <results> -i <infile>"

  exit 1;
}

GetResults() {
  IFS=' ' read -ra line <<< ${2}
  echo $line
  python -m get_beam_shift_results -i ${1} -p ${line[1]} -m ${line[2]} -o ${line[0]} --post
}

GetDir() {
  IFS=' ' read -ra line <<< ${1}
  echo "${line[0]}"
}

while getopts "i:r:h" option
do 
    case "${option}"
        in
        i) infile=${OPTARG};;
        r) results=${OPTARG};;
        h)
          Help
          ;;
    esac
done

echo "Reading from $infile"
echo "results $results"
while IFS= read -r line; do

  if [[ $line == *"#"* ]]; then
    continue
  fi

  #thisdir=`GetDir "$line"`
  GetResults "$results" "$line"
done < "$infile"
