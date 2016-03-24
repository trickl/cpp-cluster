set boxwidth 1
set xrange [0:*]
set yrange [0:*]
set xlabel "rank"
set ylabel "frequency"
set datafile separator ","
set title "Rank frequency histogram"
plot "cpmds-rankhist.dat" with boxes lw 2
pause -1 "\nPush 'q' and 'return' to exit Gnuplot ...\n"
