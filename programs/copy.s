# copy.s: host rows 0-1 into Unified Buffer rows 0-1, then back out to host rows 4-5, at 22 bytes per cycle.

.host 0
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16

.host 1
-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16

Read_Host_Memory host=0 ub=0 rows=2
Write_Host_Memory ub=0 host=4 rows=2
Halt
