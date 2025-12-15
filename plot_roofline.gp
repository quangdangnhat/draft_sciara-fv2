load "./profiling_results/roofline_specs.gp"

set title "Roofline Model (FP64)" font ",16"
set xlabel "Arithmetic Intensity (FLOP/Byte)" 
set ylabel "Performance (GFLOP/s)"
set logscale xy
set xrange [0.0001:10]
set yrange [0.09:peak_flops * 1.6]
set grid xtics ytics mxtics mytics lc rgb "#bbbbbb" lw 1 lt 2
set key bottom right

# --- Roofline Functions ---
roof(bw, x) = (bw * x < peak_flops) ? bw * x : peak_flops

# --- Styles for roofline lines ---
set style line 1 lc rgb '#8B0000' lw 2.5  # Dark Red for Roofs

# --- Color/symbol for each version ---
set style line 10 lc rgb '#8B0000' pt 7 ps 1.2   # Global - circle (dark red)
set style line 11 lc rgb '#FF6347' pt 9 ps 1.2   # Cfame - triangle (tomato)
set style line 12 lc rgb '#4169E1' pt 13 ps 1.2  # Cfamo - diamond (royal blue)
set style line 13 lc rgb '#32CD32' pt 5 ps 1.2   # Tiled - square (lime green)
set style line 14 lc rgb '#FFD700' pt 11 ps 1.2  # Tiled Halo - star (gold)

# --- Labels on Roofline lines ---
angle = 45
lx_dram = 0.008
lx_l1 = 0.0009
lx_shared = 0.003
set label 1 gprintf("DRAM: %.0f GB/s", bw_dram)   at lx_dram, roof(bw_dram, lx_dram)*1.2   tc rgb '#000000' rotate by angle font ",12"
set label 2 gprintf("L1/Tex: %.0f GB/s", bw_l1)   at lx_l1, roof(bw_l1, lx_l1)*1.2     tc rgb '#000000' rotate by angle font ",12"
set label 3 gprintf("Shared: %.0f GB/s", bw_shared) at lx_shared, roof(bw_shared, lx_shared)*1.2 tc rgb '#000000' rotate by angle font ",12"
set label 4 gprintf("Peak FP64: %.0f GFLOP/s", peak_flops) at 1, peak_flops*1.1 tc rgb '#000000' font ",12"

# --- Determine version from row and apply style ---
# Helper function to select line style based on version string
get_ls(version) = (version eq "Global" ? 10 : version eq "Cfame" ? 11 : version eq "Cfamo" ? 12 : version eq "Tiled" ? 13 : 14)

# --- Plot ---
# PNG output
set terminal png size 1200,800 enhanced font 'Arial,12'
set output './profiling_results/roofline_fp64.png'

plot \
    roof(bw_dram, x)    ls 1 lw 2 title "DRAM Roofline", \
    roof(bw_l1, x)      ls 1 lw 2 title "L1 Roofline", \
    roof(bw_shared, x)  ls 1 lw 2 title "Shared Roofline", \
    'profiling_results/roofline_data.dat' using (stringcolumn(4) eq "Global"       ? $2 : 1/0):3 with points ls 10 title "Global", \
    '' using (stringcolumn(4) eq "Cfame"        ? $2 : 1/0):3 with points ls 11 title "Cfame", \
    '' using (stringcolumn(4) eq "Cfamo"        ? $2 : 1/0):3 with points ls 12 title "Cfamo", \
    '' using (stringcolumn(4) eq "Tiled"        ? $2 : 1/0):3 with points ls 13 title "Tiled", \
    '' using (stringcolumn(4) eq "Tiled Halo"   ? $2 : 1/0):3 with points ls 14 title "Tiled Halo", \
    '' using 2:3:1 with labels center offset 0,0.8 font ",9" notitle

# SVG output
set terminal svg size 1200,800 enhanced font 'Arial,12'
set output './profiling_results/roofline_fp64.svg'

replot
