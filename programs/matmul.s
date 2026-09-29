# matmul.s: two input rows times a weight tile that copies columns 0-3 and puts twice their sum in column 4.

.host 0
1 2 3 4
.host 1
-1 0 2 5

.weights 0 0
1 0 0 0 2
.weights 0 1
0 1 0 0 2
.weights 0 2
0 0 1 0 2
.weights 0 3
0 0 0 1 2

Read_Host_Memory host=0 ub=0 rows=2
Read_Weights tile=0
MatrixMultiply ub=0 acc=0 rows=2 accumulate=0 new_weights=1
Halt
