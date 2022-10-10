file=${1};
shuf ${file} -o ${file};
split -a 2 -d -l 1000000 ${file} trainset