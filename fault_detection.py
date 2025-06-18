from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import qiskit.circuit.library as qGate
from qiskit.circuit.library import UnitaryGate
import numpy as np
from qiskit import qasm3

from qatg import QATG
from qatg import QATGFault

import sys
import os.path as osp
sys.path.append(osp.dirname(osp.abspath(__file__)))

from fault_simulation import fault_simulation
from tabulate import tabulate

from qatg import QATG
from qatg import QATGFault

import numpy as np
from Fault_model import myFault_1, myFault_2, myFault_3
from scipy.stats import chi2, ncx2
from subprocess import run, PIPE
import sys
from scipy.stats import ttest_rel, ks_2samp, mannwhitneyu

cut = "CUTs/backend_1.pyc"
qc_name = "benchmarks/qc1.qasm"

def counts_to_np(counts, num_qubits):
    keys = [format(i, f'0{num_qubits}b') for i in range(2**num_qubits)]
    arr = np.array([counts.get(k, 0) for k in keys])
    return arr

def paired_ttest(fault_model, qc, alpha=0.05):
    fault_OPD = fault_simulation(fault_model, qc, 100000)
    num_qubits = qc.num_qubits
    pA = counts_to_np(fault_simulation(None, qc, 100000), num_qubits)
    alpha = 0.2
    beta = 0.05
    k = 2**num_qubits
    df = k - 1
    pN = counts_to_np(fault_OPD, num_qubits)

    pA = pA / np.sum(pA)
    pN = pN / np.sum(pN)
    w = np.sqrt(np.sum(((pA - pN)**2) / pN))
    xT = chi2.ppf(alpha, df)
    w = 0.8 if w > 0.8 else w
    R = np.ceil(xT / w**2)
    cc = chi2.ppf(1 - alpha, df)
    non_centrality_fine = w**2 * R
    cn_fine = ncx2.ppf(beta, df, nc=non_centrality_fine)
    
    while (cn_fine < cc):
        non_centrality_fine = w**2 * R
        cn_fine = ncx2.ppf(beta, df, nc=non_centrality_fine)
        if cn_fine >= cc:
            break
        R += 1
    distribution = get_OPD(qc_name, int(R))
    p_cut = counts_to_np(distribution, num_qubits)
    pA = counts_to_np(fault_simulation(None, qc, R), num_qubits)
    statistic, p_value = mannwhitneyu(pA, p_cut, alternative='two-sided')
    # statistic, p_value = ks_2samp(pA, p_cut)
    # statistic, p_value = ttest_rel(pA, p_cut)
    if p_value < alpha:
        return True
    else:
        return False

def fault_detection (fault_model, qc, distribution=None, maximum_test_escape=None, maximum_overkill=None):
    fault_OPD = fault_simulation(fault_model, qc, 100000)
    num_qubits = qc.num_qubits
    pA = counts_to_np(fault_simulation(None, qc, 100000), num_qubits)
    # print(pA)
    # print(maximum_overkill, maximum_test_escape)
    if maximum_overkill is None:
        alpha = 0.2
    else:
        alpha = maximum_overkill
    if maximum_test_escape is None:
        beta = 0.05
    else:
        beta = maximum_test_escape
    # --- Part 1: Determine Test Repetition (R) and Critical Value (cc) ---
    k = 2**num_qubits
    df = k - 1
    pN = counts_to_np(fault_OPD, num_qubits)

    pA = pA / np.sum(pA)
    pN = pN / np.sum(pN)
    w = np.sqrt(np.sum(((pA - pN)**2) / pN))
    xT = chi2.ppf(alpha, df)
    w = 0.8 if w > 0.8 else w
    R = np.ceil(xT / w**2)
    cc = chi2.ppf(1 - alpha, df)
    non_centrality_fine = w**2 * R
    cn_fine = ncx2.ppf(beta, df, nc=non_centrality_fine)
    
    while (cn_fine < cc):
        non_centrality_fine = w**2 * R
        cn_fine = ncx2.ppf(beta, df, nc=non_centrality_fine)
        if cn_fine >= cc:
            break
        R += 1
        
    # Part 2: Simulate the CUT R
    if distribution is None:
        distribution = get_OPD(qc_name, int(R))
    p_cut = counts_to_np(distribution, num_qubits)

    # Part 3: Calculate Chi-Square Statist'ic and Make a Decision
    pA = counts_to_np(fault_simulation(None, qc, R), num_qubits)
    mask = pA != 0
    # print(p_cut)
    # print(pA)
    chi_squared_statistic = np.sqrt(np.sum(((pA[mask] - pN[mask]) ** 2) / pA[mask]))
    if chi_squared_statistic > cc:
        return False
    else:
        return True

def get_OPD(qasm_file_name, shots):
    out = run(['/home/soul/miniforge3/envs/test/bin/python', cut, qasm_file_name, str(shots)], stdout=PIPE)
    return eval(out.stdout)

if __name__ == '__main__':
    qc1 = QuantumCircuit.from_qasm_file("benchmarks/qc1.qasm")
    qc2 = QuantumCircuit.from_qasm_file("benchmarks/qc2.qasm")
    backends = ["CUTs/backend_1.pyc", "CUTs/backend_2.pyc", "CUTs/backend_3.pyc", "CUTs/backend_4.pyc"]
    fault_models = [myFault_1(), myFault_2(np.pi), myFault_3()]
    fault_labels = ["Test Configuration 1", "Test Configuration 2", "Test Configuration 3"]

    # Prepare results for both circuits
    results_qc1 = []
    results_qc2 = []

    for qc, results in zip([qc1, qc2], [results_qc1, results_qc2]):
        for cut in backends:
            row = [cut.replace("CUTs/", "").replace(".pyc", "")]
            for fault_model in fault_models:
                # result = fault_detection(fault_model, qc)
                result = paired_ttest(fault_model, qc)
                row.append("Failed" if result else "Passed")
            results.append(row)

    headers = ["Quantum Backend Simulator"] + fault_labels

    print("Results for qc1.qasm:")
    print(tabulate(results_qc1, headers=headers, tablefmt="github"))
    print("\nResults for qc2.qasm:")
    print(tabulate(results_qc2, headers=headers, tablefmt="github"))

   