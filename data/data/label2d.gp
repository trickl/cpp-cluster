set  autoscale                        # scale axes automatically
set datafile separator ","
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set title "Labelled 2D Data Plot"
plot "sample_mds_label.dat" using 1:2:3 with labels font "arial,9" tc lt 2 point pt 19 ps 1
pause -1 "\nPush 'q' and 'return' to exit Gnuplot ...\n"
