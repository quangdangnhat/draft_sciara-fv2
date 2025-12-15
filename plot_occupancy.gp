set title "GPU Occupancy" font ",16"
set xlabel "Version"
set ylabel "Occupancy (0-1)"

set style fill solid 0.6
set boxwidth 0.6
set grid ytics
set yrange [0:1]
set key off

# PNG output
set terminal png size 1200,600 enhanced font 'Arial,12'
set output './profiling_results/occupancy.png'
plot 'profiling_results/occupancy_data.dat' using 2:xtic(1) with boxes lc rgb '#1f77b4' title 'Occupancy'

# SVG output
set terminal svg size 1200,600 enhanced font 'Arial,12'
set output './profiling_results/occupancy.svg'
replot
