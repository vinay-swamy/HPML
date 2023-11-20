 #!/bin/bash

outstr=""
values=(1 5 10 50 100)

for val in "${values[@]}"; do
    output=$(./q1 "$val")
    outstr="${outstr}Q1,$val,,$output\n"
done

printf "$outstr"

