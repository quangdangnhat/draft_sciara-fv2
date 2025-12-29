import os
import re
import csv
import glob
import statistics

# Output files
RESULTS_DIR = "./profiling_results"
ROOFLINE_DAT = os.path.join(RESULTS_DIR, "roofline_data.dat")
TIME_DAT = os.path.join(RESULTS_DIR, "time_data.dat")
OCC_DAT = os.path.join(RESULTS_DIR, "occupancy_data.dat")
BENCH_FILE = os.path.join(RESULTS_DIR, "gpumembench.log")
SPECS_FILE = os.path.join(RESULTS_DIR, "roofline_specs.gp")
CONFIG_FILE = os.path.join(RESULTS_DIR, "profiling_config.txt")

# Default GPU specs (can be overridden by gpumembench log)
SPECS = {
    'bw_dram': 224.3,
    'bw_l1': 28008.7,
    'bw_shared': 2119.7,
    'peak_flops': 155.7
}

# Profiling config (scale factor for FLOPs/bytes when PROFILING_STEPS != STEPS)
PROFILING_CONFIG = {
    'steps': 16000,
    'profiling_steps': 16000,
    'scale_factor': 1.0
}

def parse_profiling_config():
    """Parse profiling_config.txt to get scale factor for FLOPs/bytes"""
    if not os.path.exists(CONFIG_FILE):
        print(f"Warning: {CONFIG_FILE} not found. Using scale_factor=1.0")
        return

    with open(CONFIG_FILE, 'r') as f:
        for line in f:
            if '=' in line:
                key, val = line.strip().split('=', 1)
                key = key.strip().upper()
                try:
                    if key == 'STEPS':
                        PROFILING_CONFIG['steps'] = int(val)
                    elif key == 'PROFILING_STEPS':
                        PROFILING_CONFIG['profiling_steps'] = int(val)
                    elif key == 'SCALE_FACTOR':
                        PROFILING_CONFIG['scale_factor'] = float(val)
                except:
                    pass

    # Calculate scale factor if not explicitly set
    if PROFILING_CONFIG['profiling_steps'] > 0:
        PROFILING_CONFIG['scale_factor'] = PROFILING_CONFIG['steps'] / PROFILING_CONFIG['profiling_steps']

    print(f"Profiling Config: STEPS={PROFILING_CONFIG['steps']}, PROFILING_STEPS={PROFILING_CONFIG['profiling_steps']}, scale_factor={PROFILING_CONFIG['scale_factor']:.2f}x")

def parse_gpumembench():
    """Parses gpumembench.log for empirical bandwidths"""
    if not os.path.exists(BENCH_FILE):
        print(f"Warning: {BENCH_FILE} not found. Using defaults.")
        return

    with open(BENCH_FILE, 'r') as f:
        content = f.read()

    dram = re.search(r'Global.*?read.*?:.*?([0-9\.]+)\s*GB/s', content, re.IGNORECASE)
    shared = re.search(r'Shared.*?read.*?:.*?([0-9\.]+)\s*GB/s', content, re.IGNORECASE)
    l2 = re.search(r'Texture.*?read.*?:.*?([0-9\.]+)\s*GB/s', content, re.IGNORECASE)
    peak_flops = re.search(r'Peak FP64.*:.*?([0-9\.]+)\s*GFLOP/s', content, re.IGNORECASE)

    if dram: SPECS['bw_dram'] = float(dram.group(1))
    if shared: SPECS['bw_shared'] = float(shared.group(1))
    if l2: SPECS['bw_l1'] = float(l2.group(1))
    if peak_flops: SPECS['peak_flops'] = float(peak_flops.group(1))
    
    print(f"Specs Loaded: DRAM={SPECS['bw_dram']} GB/s, L1={SPECS['bw_l1']} GB/s, Shared={SPECS['bw_shared']} GB/s, Peak GFLOPS={SPECS['peak_flops']} GFLOP/s")

TRANS_SIZE = 32.0

def get_unit_multiplier(unit_str):
    """Returns the multiplier to convert the unit to Seconds."""
    if not unit_str: return 1.0
    u = unit_str.lower().strip()
    if u == 's': return 1.0
    if u == 'ms': return 1e-3
    if u == 'us': return 1e-6
    if u == 'ns': return 1e-9
    return 1.0

def read_csv_with_units(path):
    """
    Reads CSV, detecting if a unit row exists immediately after header.
    Returns a list of dicts, where values are converted to standard units (seconds) if applicable.
    """
    rows = []
    if not os.path.exists(path):
        return rows
        
    with open(path, 'r', newline='') as f:
        lines = f.readlines()

    # 1. Locate Header Line
    header_idx = -1
    for i, L in enumerate(lines[:20]):
        if '"Name"' in L or 'Name' in L:
            header_idx = i
            break
    
    if header_idx == -1: return rows

    # 2. Parse Header
    keys = [k.strip().replace('"', '') for k in lines[header_idx].strip().split(',')]
    
    # 3. Check for Unit Line (immediate next line)
    # Unit lines usually look like: ,%,s,,us,us,us,
    unit_map = {}
    data_start_idx = header_idx + 1
    
    if len(lines) > header_idx + 1:
        next_line = lines[header_idx + 1]
        potential_units = [u.strip().replace('"', '') for u in next_line.strip().split(',')]

        if 's' in potential_units or 'us' in potential_units or 'ms' in potential_units:
            data_start_idx = header_idx + 2 # Skip unit line for data
            for i, u in enumerate(potential_units):
                if i < len(keys):
                    # Store multiplier for this column name
                    unit_map[keys[i]] = get_unit_multiplier(u)

    # 4. Read Data
    reader = csv.DictReader(lines[data_start_idx:], fieldnames=keys)
    for r in reader:
        # Apply multipliers if they exist
        clean_row = r.copy()
        for k, v in r.items():
            if not v: continue
            if k in unit_map and unit_map[k] != 1.0:
                val_clean = re.sub(r'[^0-9\.]', '', v)
                try:
                    clean_row[k] = float(val_clean) * unit_map[k]
                except:
                    clean_row[k] = v
        rows.append(clean_row)
        
    return rows

def parse_elapsed_from_logs():
    # Look for profiling_results/*.log and lines like: Elapsed time [s]: 12.34
    times = {}
    for p in glob.glob(os.path.join(RESULTS_DIR, '*.log')):
        name = os.path.basename(p).split('.')[0]
        with open(p, 'r') as f:
            for L in f:
                m = re.search(r'Elapsed time \[s\]:\s*([0-9\.]+)', L)
                if m:
                    times[name] = float(m.group(1))
    return times

def clean_version(base_name):
    s = base_name.replace('sciara_cuda_', '').replace('sciara_', '')
    s = re.sub(r'[_\-]+', ' ', s).strip()
    s = s.title()
    if s.lower() == 'cuda' or s.lower() == 'global':
        return 'Global'
    return s

def match_kernel_name(name):
    """Match kernel name to a canonical name for grouping"""
    if not name: return None
    # Standard kernels
    if 'computeOutflows' in name: return 'computeOutflows'
    if 'massBalance' in name: return 'massBalance'
    # CfA combined kernels (these combine outflows + massBalance)
    if 'CfA_Me' in name: return 'CfA_Me'
    if 'CfA_Mo' in name: return 'CfA_Mo'
    # Additional kernels that contribute to computation
    if 'initBuffers' in name: return 'initBuffers'
    if 'normalizeTemperature' in name: return 'normalizeTemperature'
    return None

def parse_one_dataset(base):
    data = {}
    compute_file = os.path.join(RESULTS_DIR, base + '_compute.csv')
    mem_file = os.path.join(RESULTS_DIR, base + '_memory.csv')
    sum_file = os.path.join(RESULTS_DIR, base + '_gpu_summary.csv')
    occ_file = os.path.join(RESULTS_DIR, base + '_occupancy.csv')

    comp_rows = read_csv_with_units(compute_file)
    mem_rows = read_csv_with_units(mem_file)
    sum_rows = read_csv_with_units(sum_file)
    occ_rows = read_csv_with_units(occ_file)

    kernels = {}
    
    def add_kernel(k):
        if k not in kernels:
            kernels[k] = {
                'total_flops': 0.0,
                'invocations': 0,
                'total_bytes': 0.0,
                'occupancies': [],
                'time_total_s': 0.0
            }

    for r in sum_rows:
        k = match_kernel_name(r.get('Name', ''))
        if not k: continue
        add_kernel(k)
        
        if 'Time' in r and r['Time']:
            try:
                kernels[k]['time_total_s'] = float(r['Time'])
            except:
                pass
        
        if kernels[k]['time_total_s'] == 0.0 and 'Avg' in r:
            try:
                t_avg = float(r['Avg'])
                inv_str = str(r.get('Invocations', '1'))
                inv = int(re.sub(r'[^0-9]', '', inv_str) or 1)
                kernels[k]['time_total_s'] = t_avg * inv
                kernels[k]['invocations'] = inv
            except:
                pass
        
        if 'Invocations' in r and kernels[k]['invocations'] == 0:
            try:
                inv_str = str(r['Invocations'])
                kernels[k]['invocations'] = int(re.sub(r'[^0-9]', '', inv_str) or 1)
            except:
                pass

    for r in comp_rows:
        k = match_kernel_name(r.get('Kernel', r.get('Name', '')))
        if not k: continue
        add_kernel(k)
        
        metric = r.get('Metric Name', '')
        val_s = str(r.get('Metric Value', r.get('Avg', '0')))
        
        try:
            val = float(re.sub(r'[^0-9\.]', '', val_s))
        except:
            val = 0.0

        if 'Invocations' in r and kernels[k]['invocations'] == 0:
            inv_s = str(r.get('Invocations', '1'))
            try:
                kernels[k]['invocations'] = int(re.sub(r'[^0-9]', '', inv_s) or 1)
            except:
                pass

        # Apply scale factor to FLOPs (profiling may use fewer steps than benchmark)
        # val is the Average FLOPs per invocation, so we need to multiply by invocations
        scale = PROFILING_CONFIG['scale_factor']
        invocations = kernels[k].get('invocations', 1)
        if invocations == 0:
            invocations = 1

        if metric == 'flop_count_dp':
            kernels[k]['total_flops'] += val * invocations * scale
        elif metric == 'flop_count_sp':
            kernels[k]['total_flops'] += val * invocations * scale
        elif 'flop' in metric.lower():
            kernels[k]['total_flops'] += val * invocations * scale

    for r in mem_rows:
        k = match_kernel_name(r.get('Kernel', r.get('Name', '')))
        if not k: continue
        add_kernel(k)

        def getval(f):
            return float(re.sub(r'[^0-9\.]', '', str(r.get(f, 0))))

        trans = getval('Avg')

        # Apply scale factor to bytes (profiling may use fewer steps than benchmark)
        # trans is the Average transactions per invocation, so we need to multiply by invocations
        scale = PROFILING_CONFIG['scale_factor']
        invocations = kernels[k].get('invocations', 1)
        if invocations == 0:
            invocations = 1
        kernels[k]['total_bytes'] += trans * TRANS_SIZE * invocations * scale

    for r in occ_rows:
        k = match_kernel_name(r.get('Kernel', r.get('Name', '')))
        if not k: continue
        add_kernel(k)
        val_str = str(r.get('Metric Value', r.get('Avg', '0')))
        try:
            val = float(re.sub(r'[^0-9\.]', '', val_str))
            kernels[k]['occupancies'].append(val)
        except:
            pass

    data['kernels'] = kernels
    return data

def calc_roofline_point(total_flops, total_bytes, time_s, version_name, kernel_name=""):
    """
    Calculate a point for the roofline plot.
    Returns: (gflops, ai) or None if data is invalid
    """
    if time_s <= 0 or total_flops <= 0:
        print(f"  Warning [{version_name}{' - ' + kernel_name if kernel_name else ''}]: Invalid data (time={time_s:.6f}s, flops={total_flops:.2e})")
        return None

    # GFLOPS = FLOP total / time total / 1e9 (NOT 1e5!)
    # 1 GFLOPS = 10^9 FLOPS
    gflops = (total_flops / time_s) / 1e9

    # Arithmetic Intensity = FLOP / Byte
    ai = total_flops / max(1.0, total_bytes)

    # Debug output
    print(f"  [{version_name}{' - ' + kernel_name if kernel_name else ''}]:")
    print(f"    FLOP total: {total_flops:.2e}")
    print(f"    Time: {time_s:.6f} s")
    print(f"    GFLOPS: {gflops:.4f}")
    print(f"    Total bytes: {total_bytes:.2e} -> AI: {ai:.6f} FLOP/byte")

    return gflops, ai

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Parse profiling config first to get scale factor
    parse_profiling_config()
    parse_gpumembench()

    # Write specs file for gnuplot
    with open(SPECS_FILE, 'w') as f:
        f.write(f"bw_dram = {SPECS['bw_dram']}\n")
        f.write(f"bw_l1 = {SPECS['bw_l1']}\n")
        f.write(f"bw_shared = {SPECS['bw_shared']}\n")
        f.write(f"peak_flops = {SPECS['peak_flops']}\n")

    compute_files = glob.glob(os.path.join(RESULTS_DIR, '*_compute.csv'))

    roofline_rows = []
    time_map = {}
    occ_map = {}
    log_times = parse_elapsed_from_logs()

    print("\n=== Processing datasets ===")
    
    for comp in compute_files:
        base = os.path.basename(comp).replace('_compute.csv', '')
        version = clean_version(base)
        
        print(f"\n--- {version} ({base}) ---")
        
        ds = parse_one_dataset(base)
        kernels = ds['kernels']

        # Elapsed time: prefer log, else sum kernel times from gpu_summary
        if base in log_times:
            time_map[version] = log_times[base]
        else:
            total_time = sum(k['time_total_s'] for k in kernels.values())
            time_map[version] = total_time

        # Occupancy: average of all main kernels
        occ_vals = []
        for k_name in ('massBalance', 'computeOutflows', 'CfA_Me', 'CfA_Mo', 'initBuffers', 'normalizeTemperature'):
            if k_name in kernels and kernels[k_name]['occupancies']:
                occ_vals.append(statistics.mean(kernels[k_name]['occupancies']))
        occ_map[version] = statistics.mean(occ_vals) if occ_vals else 0.0

        # Roofline calculation: combine relevant kernels based on version type
        k_mass = kernels.get('massBalance')
        k_outflows = kernels.get('computeOutflows')
        k_cfame = kernels.get('CfA_Me')
        k_cfamo = kernels.get('CfA_Mo')
        k_init = kernels.get('initBuffers')
        k_norm = kernels.get('normalizeTemperature')

        roofline_point = None

        # For standard Global/Tiled versions: combine massBalance + computeOutflows
        if k_mass and k_outflows and (k_mass['total_flops'] > 0 or k_outflows['total_flops'] > 0):
            combined_flops = k_mass['total_flops'] + k_outflows['total_flops']
            combined_total_bytes = k_mass['total_bytes'] + k_outflows['total_bytes']
            combined_time = k_mass['time_total_s'] + k_outflows['time_total_s']

            roofline_point = calc_roofline_point(
                combined_flops, combined_total_bytes,
                combined_time, version, "massBalance + computeOutflows"
            )
        # For CfAMe version: use CfA_Me kernel + helper kernels
        elif k_cfame:
            combined_flops = k_cfame['total_flops']
            combined_total_bytes = k_cfame['total_bytes']
            combined_time = k_cfame['time_total_s']

            # Add initBuffers and normalizeTemperature if present
            if k_init:
                combined_flops += k_init['total_flops']
                combined_total_bytes += k_init['total_bytes']
                combined_time += k_init['time_total_s']
            if k_norm:
                combined_flops += k_norm['total_flops']
                combined_total_bytes += k_norm['total_bytes']
                combined_time += k_norm['time_total_s']

            if combined_flops > 0:
                roofline_point = calc_roofline_point(
                    combined_flops, combined_total_bytes,
                    combined_time, version, "CfA_Me + helpers"
                )
        # For CfAMo version: use CfA_Mo kernel + helper kernels
        elif k_cfamo:
            combined_flops = k_cfamo['total_flops']
            combined_total_bytes = k_cfamo['total_bytes']
            combined_time = k_cfamo['time_total_s']

            # Add initBuffers and normalizeTemperature if present
            if k_init:
                combined_flops += k_init['total_flops']
                combined_total_bytes += k_init['total_bytes']
                combined_time += k_init['time_total_s']
            if k_norm:
                combined_flops += k_norm['total_flops']
                combined_total_bytes += k_norm['total_bytes']
                combined_time += k_norm['time_total_s']

            if combined_flops > 0:
                roofline_point = calc_roofline_point(
                    combined_flops, combined_total_bytes,
                    combined_time, version, "CfA_Mo + helpers"
                )
        
        if roofline_point:
            gflops, ai = roofline_point
            roofline_rows.append({
                'label': version,
                'ai': ai,
                'gflops': gflops,
                'version': version
            })
        else:
            print(f"  Warning: No valid roofline data for {version}")

    # Write output files
    # Note: Use underscores instead of spaces for version names to avoid gnuplot parsing issues
    with open(ROOFLINE_DAT, 'w') as f:
        f.write('# Label AI GFLOPS Version\n')
        for r in roofline_rows:
            # Replace spaces with underscores for gnuplot compatibility
            label_safe = r["label"].replace(" ", "_")
            version_safe = r["version"].replace(" ", "_")
            f.write(f'{label_safe} {r["ai"]:.6f} {r["gflops"]:.4f} {version_safe}\n')

    with open(TIME_DAT, 'w') as f:
        f.write('# Version Time_s\n')
        for v, t in sorted(time_map.items(), key=lambda x: x[1], reverse=True):
            # Replace spaces with underscores for gnuplot compatibility
            v_safe = v.replace(" ", "_")
            f.write(f'{v_safe} {t:.6f}\n')

    with open(OCC_DAT, 'w') as f:
        f.write('# Version Occupancy(0-1)\n')
        for v, o in sorted(occ_map.items(), key=lambda x: x[0]):
            # Replace spaces with underscores for gnuplot compatibility
            v_safe = v.replace(" ", "_")
            f.write(f'{v_safe} {o:.6f}\n')

    print(f'\n=== Output files written ===')
    print(f'Roofline data: {ROOFLINE_DAT}')
    print(f'Time data: {TIME_DAT}')
    print(f'Occupancy data: {OCC_DAT}')
    print(f'Specs file: {SPECS_FILE}')

if __name__ == '__main__':
    main()