# weights.s: queue two tiles; each takes 1366 cycles to arrive at 48 bytes per cycle. Add three more Read_Weights to see the FIFO fill.

.weights 0
1 2 3 4 5 6 7 8

.weights 1
-1 -2 -3 -4 -5 -6 -7 -8

Read_Weights tile=0
Read_Weights tile=1
Halt
