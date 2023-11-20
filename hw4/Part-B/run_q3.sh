 #!/bin/bash

gridwidth=$1
blockwidth=$2


outstr=""
values=(1 5 10 50 100)
mil=1000000
for val in "${values[@]}"; do
    if [ "$gridwidth" -eq -1 ]; then
        n_elem=$((mil * val))
        gridwidth=$((n_elem / blockwidth))
        output=$(./q2 "$gridwidth" "$blockwidth" "$val")    
        outstr="${outstr}Q3,$val,${gridwidth}_${blockwidth},$output\n"
        gridwidth=-1
    else
        
        output=$(./q3 "$gridwidth" "$blockwidth" "$val")    
        outstr="${outstr}Q3,$val,${gridwidth}_${blockwidth},$output\n"
    fi
    
done

printf "$outstr" 
