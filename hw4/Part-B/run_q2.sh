 #!/bin/bash

gridwidth=$1
blockwidth=$2


outstr=""
values=(1 5 10 50 100)

for val in "${values[@]}"; do
    if [ "$gridwidth" -eq -1 ]; then
        n_elem=$(echo "1000000 * $val" | bc)
        gridwidth=$(echo "$n_elem / $blockwidth" | bc)
    fi
    output=$(./q2 "$gridwidth" "$blockwidth" "$val")
    outstr="$outstr$output,"
done

echo "${outstr::-1}"
