for i in 42
do
python main_Fermi.py -epochs 10000 -runs 1 \
    -L_num 200 -question 1 -is_cc -seed ${i}
done
