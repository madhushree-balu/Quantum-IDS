"""
Quantum IDS Project - Run All Scripts
File: run_all.py
Purpose: Execute the complete analysis pipeline
"""

import subprocess
import sys
import time
import os

def run_script(script_path, script_name):
    """Run a Python script and handle errors"""
    print("\n" + "="*80)
    print(f"RUNNING: {script_name}")
    print("="*80)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False,
            text=True,
            check=True
        )
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ {script_name} completed successfully in {elapsed_time:.2f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ {script_name} failed after {elapsed_time:.2f}s")
        print(f"Error: {e}")
        return False

def main():
    print("="*80)
    print("QUANTUM IDS PROJECT - COMPLETE PIPELINE EXECUTION")
    print("="*80)
    print("\nThis will run all analysis scripts in sequence:")
    print("  1. Data Preprocessing")
    print("  2. Classical Baseline Models")
    print("  3. Quantum Kernel Implementation")
    print("  4. Comparative Analysis")
    print("\nEstimated total time: 15-45 minutes (depending on hardware)")
    print("="*80)
    
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Define scripts to run
    scripts = [
        ("src/01_preprocessing.py", "Data Preprocessing"),
        ("src/02_classical_baseline.py", "Classical Baseline"),
        ("src/03_quantum_kernel.py", "Quantum Kernel Implementation"),
        ("src/04_comparative_analysis.py", "Comparative Analysis"),
    ]
    
    # Track overall progress
    total_start_time = time.time()
    results = []
    
    # Run each script
    for script_path, script_name in scripts:
        if not os.path.exists(script_path):
            print(f"\n✗ Script not found: {script_path}")
            print(f"  Make sure all scripts are in the 'src/' directory")
            results.append(False)
            break
            
        success = run_script(script_path, script_name)
        results.append(success)
        
        if not success:
            print(f"\n✗ Pipeline stopped due to error in {script_name}")
            break
    
    # Summary
    total_time = time.time() - total_start_time
    
    print("\n" + "="*80)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*80)
    
    for (script_path, script_name), success in zip(scripts, results):
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {script_name}")
    
    print(f"\nTotal execution time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    
    if all(results):
        print("\n✓ All scripts completed successfully!")
        print("\nNext steps:")
        print("  1. Review results in 'results/' directory")
        print("  2. Check figures in 'results/figures/' and 'paper/figures/'")
        print("  3. Read 'results/final_report.txt' for summary")
        print("  4. Use paper figures for your manuscript")
    else:
        print("\n✗ Pipeline completed with errors. Check logs above.")
    
    print("="*80)

if __name__ == "__main__":
    main()