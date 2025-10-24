import subprocess
import os
import shutil

def get_home_directory():
    return os.path.expanduser("~")

def filter_labels(label_dir, allowed_classes):
    """
    Removes lines from YOLO label files that are not in allowed_classes.
    label_dir: path containing *.txt YOLO labels
    allowed_classes: list of int IDs to keep
    """
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            file_path = os.path.join(label_dir, label_file)
            with open(file_path, "r") as f:
                lines = f.readlines()
            filtered_lines = [
                line for line in lines
                if line.strip() and int(line.split()[0]) in allowed_classes
            ]
            with open(file_path, "w") as f:
                f.writelines(filtered_lines)

def create_python_env_and_run_script():
    env_name = 'Pythontorchenv'
    home_directory = get_home_directory()
    envpath = f"{home_directory}/{env_name}"

    # ✅ Allowed YOLO classes (match to your dataset.yaml mapping)
    allowed_classes = [0]
    label_dir = f"{home_directory}/Akhil_yolo/Research/BraTS21_preprocessed2/val"  # adjust if needed
    filter_labels(label_dir, allowed_classes)

    # Create venv if missing
    if not os.path.exists(envpath):
        subprocess.run(['python3.9', '-m', 'venv', envpath])

    # Install requirements
    install_polars_cmd = f'{envpath}/bin/pip install --upgrade polars'
    subprocess.run(install_polars_cmd, shell=True)

    # Run training
    run_script_cmd = f'{envpath}/bin/python3.9 {home_directory}/Akhil_yolo/Research/trainm.py'
    subprocess.run(run_script_cmd, shell=True)


if __name__ == '__main__':
    create_python_env_and_run_script()
