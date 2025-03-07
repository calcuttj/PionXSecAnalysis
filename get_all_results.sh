#!/bin/bash

Help() {
  echo "get_all_results.sh -i <infile> -n <npars>"

  exit 1;
}

GetResults() {
  IFS=' ' read -ra line <<< ${1}
  echo $line
  cd ${line[0]}
  python -m fit_results --no_michel -i ${line[1]} --nothrows --noplotstyle -o results.root
  python -m fit_results --no_michel -i results.root --routine save -o asdf
  python -m fit_results --no_michel -i ${line[1]} --routine pars --save --npars ${2}
  cd ../
}

GetDir() {
  IFS=' ' read -ra line <<< ${1}
  echo "${line[0]}"
}

GetFitFile() {
  IFS=' ' read -ra line <<< ${1}
  echo "${line[1]}"
}

npars=17
while getopts "i:n:h" option
do 
    case "${option}"
        in
        i) infile=${OPTARG};;
        n) npars=${OPTARG};;
        h)
          Help
          ;;
    esac
done

echo "Reading from $infile"
echo "npars: $npars"
while IFS= read -r line; do

  if [[ $line == *"#"* ]]; then
    continue
  fi

  thisdir=`GetDir "$line"`
  stat $thisdir
  if [ $? -ne 0 ] ; then
    mkdir $thisdir
  fi

  GetResults "$line" $npars
done < "$infile"
