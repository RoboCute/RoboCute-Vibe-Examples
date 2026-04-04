#!/usr/bin/env python3
"""Build script for RoboCute project."""

import argparse
import subprocess
import shutil
import sys
import tarfile
import gzip
from pathlib import Path


def run_command(cmd, cwd):
    """Run a command and print output."""
    print(f">>> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        sys.exit(result.returncode)


def extract_tar_gz(directory, current_dir):
    """Extract *.tar.gz to *.tar, then extract *.tar files."""
    directory = Path(directory)
    current_dir = Path(current_dir)
    tar_gz_files = list(directory.glob("*.tar.gz"))
    
    for tar_gz_path in tar_gz_files:
        tar_path = tar_gz_path.with_suffix('')  # Remove .gz to get .tar
        # Step 1: Extract .tar.gz to .tar (gunzip)
        print(f">>> Unzipping {tar_gz_path.name} to {tar_path.name}")
        with gzip.open(tar_gz_path, 'rb') as f_in:
            with open(tar_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Step 2: Extract .tar file
        print(f">>> Extracting {tar_path.name}")
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(path=directory)
        
        # Optional: Remove the intermediate .tar file
        tar_path.unlink()
        print(f"Extracted {tar_gz_path.name} successfully")
        folder_path = tar_path.with_suffix('') / 'src'
        # Copy folder_path to current dir
        if folder_path.exists():
            shutil.rmtree('src/', ignore_errors=True)
            shutil.move(folder_path, '.')
            
        


def main():
    parser = argparse.ArgumentParser(description="Build script for RoboCute project.")
    parser.add_argument(
        "work_dir",
        nargs="?",
        default="",
        help="Path to the RoboCute work directory (default: D:/RoboCute)",
    )
    parser.add_argument(
        "-r", "--rebuild",
        action="store_true",
        help="Force rebuild by passing -r to build_and_copy.py",
    )
    parser.add_argument(
        "-m", "--mode",
        choices=["release", "debug"],
        default="release",
        help="Build mode: release or debug (default: release)",
    )
    args = parser.parse_args()
    if not args.work_dir:
        print('RoboCute directory require in argument!')
        exit(1)
    work_dir = Path(args.work_dir)
    dist_dir = work_dir / "dist"
    current_dir = Path.cwd()

    # Run xmake f -m <mode> -c
    run_command(["xmake", "f", "-m", args.mode, "-c"], cwd=work_dir)

    # Run xmake
    build_cmd = ['uv', 'run', 'scripts/build_and_copy.py', '-m', args.mode]
    if args.rebuild:
        build_cmd.insert(3, '-r')
    run_command(build_cmd, cwd=work_dir)

    # Run uv build
    run_command(["uv", "build"], cwd=work_dir)

    # Copy D:/RoboCute/dist/ to current directory
    if dist_dir.exists():
        target_dir = current_dir / "dist"
        print(f">>> Copying {dist_dir} to {target_dir}")
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(dist_dir, target_dir)
        print(f"Copied successfully to {target_dir}")
        
        # Unzip *.tar.gz to *.tar, then extract *.tar
        extract_tar_gz(target_dir, current_dir)
    else:
        print(f"Warning: {dist_dir} does not exist")


if __name__ == "__main__":
    main()
