# Import stuff!
import os, sys

def make_powershell_parallel_script(electrodes = None, num_cpu = None, process_path = None, process_code = None):
    """
    electrodes:: a list of electrode numbers (int)
    """
    n_cores_to_be_used = num_cpu
    path_ = os.path.join(process_path, process_code)
    if 'umap' in process_code:
        f = open('blech_clust_spike_scatter_PS_parallel.ps1', 'w')
    else:
        f = open('blech_clust_PS_parallel.ps1', 'w')
    print("# Define the scripts and their arguments",  file = f)
    print("$scripts = @(", file = f)
    for i in electrodes:
        print('\t@{Path = "%s"; Args = @("%i")}' % (path_, i), file=f)
    print(")", file = f)
    print('\n', file = f)
    print("# Define the throttle limit",  file = f)
    print('$throttleLimit = %i' % n_cores_to_be_used, file = f)
    print('\n', file = f)
    print('# Start the jobs with throttle limit', file = f)
    print('$jobs = foreach ($script in $scripts) {', file=f)
    print('\tStart-ThreadJob -ScriptBlock {', file=f)
    print('\t\tparam ($scriptPath, $scriptArgs)', file=f)
    print('\t\tpython $scriptPath $scriptArgs', file=f)
    print('\t} -ArgumentList $script.Path, ($script.Args -join " ") -ThrottleLimit $throttleLimit', file=f)
    print('}', file=f)
    print('\n', file = f)
    print('$jobs | ForEach-Object { $_ | Wait-Job }', file=f)
    print('$jobs | ForEach-Object {', file=f)
    print('\t$output = $_ | Receive-Job', file=f)
    print('\tWrite-Output "Output from $($_.Name):"', file=f)
    print('\tWrite-Output $output', file=f)
    print('}', file=f)
    print('\n', file = f)
    print('# Clean up the jobs', file=f)
    print('$jobs | ForEach-Object { $_ | Remove-Job }', file=f)
    f.close()

    