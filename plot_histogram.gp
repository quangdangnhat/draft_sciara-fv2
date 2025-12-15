set title "Total Execution Time (16000 Steps)" font ",16"
set ylabel "Elapsed Time (s)"
set grid y
set style data histograms
set style fill solid 0.8 border -1
set boxwidth 0.6
set yrange [0:*]
set style line 1 lc rgb '#4F9FE0'

# PNG output
set terminal png size 800,600 enhanced font 'Arial,12'
set output './profiling_results/histogram_times.png'
plot './profiling_results/time_data.dat' using 2:xtic(1) ls 1 title "Execution Time", \
    '' using ($0):2:(sprintf("%.2f",$2)) with labels center offset 0,0.5 notitle

# SVG output
set terminal svg size 800,600 enhanced font 'Arial,12'
set output './profiling_results/histogram_times.svg'
plot './profiling_results/time_data.dat' using 2:xtic(1) ls 1 title "Execution Time", \
    '' using ($0):2:(sprintf("%.2f",$2)) with labels center offset 0,0.5 notitle
