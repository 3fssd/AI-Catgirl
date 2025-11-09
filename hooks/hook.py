from PyInstaller.utils.hooks import collect_data_files, get_package_paths
import os, sys
# Get the package path
package_path = get_package_paths('ctransformers')[0]
# Collect data files
datas = collect_data_files('ctransformers')
# Append the additional .dll or .so file
if os.name == 'nt':  # Windows
    dll_path = os.path.join(package_path, 'ctransformers', 'ctransformers.dll')
    datas.append((dll_path, 'ctransformers'))