import numpy as np
import qiskit.circuit.library as qGate
from qiskit.converters import circuit_to_gate
from qiskit.circuit.library import UnitaryGate
from qiskit import qasm3
from qiskit import QuantumCircuit

from qatg import QATG
from qatg import QATGFault

class myFault_1(QATGFault):
	def __init__(self):
		super(myFault_1, self).__init__(qGate.SXGate, 0, f"gateType: SX, qubits: 0")
	def createOriginalGate(self):
		return qGate.SXGate()
	def createFaultyGate(self, faultfreeGate):
		if not isinstance(faultfreeGate, qGate.SXGate):
			raise TypeError("what is this faultfreeGate")
		matrix = qGate.SXGate().to_matrix()
		RZF = qGate.RZGate(np.pi/20).to_matrix()
		return UnitaryGate(np.matmul(RZF, matrix))

class myFault_2(QATGFault):
	def __init__(self, param):
		super(myFault_2, self).__init__(qGate.RZGate, 0, f"gateType: RZ, qubits: 0, params: {param}")
		self.param = param
	def createOriginalGate(self):
		return qGate.RZGate(self.param)
	def createFaultyGate(self, faultfreeGate):
		if not isinstance(faultfreeGate, qGate.RZGate):
			raise TypeError("what is this faultfreeGate")
		matrix = qGate.RZGate(faultfreeGate.params[0]).to_matrix()
		RYF = qGate.RYGate(0.1 * faultfreeGate.params[0]).to_matrix()
		return UnitaryGate(np.matmul(RYF, matrix))

class myFault_3(QATGFault):
	def __init__(self):
		super(myFault_3, self).__init__(qGate.CXGate, [0, 1], f"gateType: CX, qubits: 0-1")
	def createOriginalGate(self):
		return qGate.CXGate()
	def createFaultyGate(self, faultfreeGate):
		if not isinstance(faultfreeGate, qGate.CXGate):
			raise TypeError("what is this faultfreeGate")
		matrix = qGate.CXGate().to_matrix()
		RXF1 = qGate.RXGate(0.1 * np.pi)
		RXF2 = qGate.RXGate(-0.1 * np.pi)
		matrix = np.matmul(matrix, np.kron(np.eye(2), RXF1))
		matrix = np.matmul(np.kron(np.eye(2), RXF2), matrix)
		return UnitaryGate(matrix)

if __name__ == '__main__':
	generator = QATG(circuitSize = 1, basisSingleQubitGateSet = [qGate.UGate], circuitInitializedStates = {1: [1, 0]}, minRequiredStateFidelity = 0.1)
	configurationList = generator.createTestConfiguration([myFault_1()])

	for idx, configuration in enumerate(configurationList):
		print(configuration)
		with open(f'Sx_test.qasm', 'w') as f:
			qasm3.dump(configuration.circuit, f)

	generator = QATG(circuitSize = 1, basisSingleQubitGateSet = [qGate.UGate], circuitInitializedStates = {1: [1, 0]}, minRequiredStateFidelity = 0.1)
	configurationList = generator.createTestConfiguration([myFault_2(np.pi)])

	for idx, configuration in enumerate(configurationList):
		print(configuration)
		with open(f'Rz_test.qasm', 'w') as f:
			qasm3.dump(configuration.circuit, f)

	generator = QATG(circuitSize = 2, basisSingleQubitGateSet = [qGate.UGate], circuitInitializedStates = {2: [1, 0, 0, 0]}, minRequiredStateFidelity = 0.1)
	configurationList = generator.createTestConfiguration([myFault_3()])

	for idx, configuration in enumerate(configurationList):
		print(configuration)
		with open(f'Cnot_test.qasm', 'w') as f:
			qasm3.dump(configuration.circuit, f)