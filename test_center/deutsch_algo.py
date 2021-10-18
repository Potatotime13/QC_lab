from cirq.circuits import circuit
import numpy as np
import cirq
import random as rn

def deutsch_algorithm(q0, q1, oracle):
    circuit = cirq.Circuit()
    circuit.append([cirq.H(q0)])
    circuit.append([cirq.X(q1), cirq.H(q1)])
    circuit.append(oracle)
    circuit.append([cirq.H(q0)])
    circuit.append(cirq.measure(q0))

    return circuit


def main():
    q0, q1 = cirq.LineQubit.range(2)
    oracles = {"null": [], "ones": [cirq.X(q1)], "x": [cirq.CNOT(q0, q1)], "notx": [cirq.CNOT(q0, q1), cirq.X(q1)]}
    sim = cirq.Simulator()
    for key, value in oracles.items():
        result = sim.run(deutsch_algorithm(q0,q1,value), repetitions=10)
        print('oracle: {:<4} results: {}'.format(key, result))


if __name__ == '__main__':
    main()