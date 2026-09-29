# activate.s: matmul.s's two rows, then ReLU at shift 1 into UB rows 2-3 and tanh at shift 2 into rows 4-5, all four back to host rows 2-5.

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
Activate acc=0 ub=2 rows=2 shift=1 function=relu
Activate acc=0 ub=4 rows=2 shift=2 function=tanh
Write_Host_Memory ub=2 host=2 rows=4
Halt
