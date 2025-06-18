from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import qiskit.circuit.library as qGate
from qiskit.circuit.library import UnitaryGate
import numpy as np

import sys
import os.path as osp
sys.path.append(osp.dirname(osp.abspath(__file__)))

from Fault_model import myFault_1, myFault_2, myFault_3
from tabulate import tabulate

def fault_simulation(fault_model, qc, shot):
    backend = AerSimulator()
    faulty_qc = qc.copy()   
    if fault_model is not None:
        for idx, gate in enumerate(faulty_qc.data):
            if fault_model.isSameGateType(gate[0]):
                if gate[0].params:
                    fault_model.param = gate[0].params[0]
                faulty_gate = fault_model.createFaultyGate(gate[0])
                faulty_qc.data[idx] = (faulty_gate, gate[1], gate[2])
    new_circuit = transpile(faulty_qc, backend)
    simulateJob = backend.run(new_circuit, shots=shot)
    result_counts = simulateJob.result().get_counts()
    return result_counts

if __name__ == '__main__':
    qc1 = QuantumCircuit.from_qasm_file("benchmarks/qc1.qasm")
    qc1_result = []
    for fault_model in [None, myFault_1(), myFault_2(np.pi), myFault_3()]:
        result_counts = fault_simulation(fault_model, qc1, 100000)
        qc1_result.append(result_counts)
    qc2 = QuantumCircuit.from_qasm_file("benchmarks/qc2.qasm")
    qc2_result = []
    for fault_model in [None, myFault_1(), myFault_2(np.pi), myFault_3()]:
        result_counts = fault_simulation(fault_model, qc2, 100000)
        qc2_result.append(result_counts)
        
    table = []
    fault_labels = ["Fault-free", "1", "2", "3"]
    for idx, (r1, r2) in enumerate(zip(qc1_result, qc2_result)):
        table.append([fault_labels[idx], str(r1), str(r2)])

    headers = ["Fault Index", "qc1.qasm result", "qc2.qasm result"]
    print(tabulate(table, headers=headers, tablefmt="github"))   