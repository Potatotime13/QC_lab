from cirq.circuits import circuit
import numpy as np
import cirq
import random as rn

def qc_teleport(ranX, ranY):
    circuit = cirq.Circuit()
    msg, alice, bob = cirq.LineQubit.range(3)

    circuit.append([cirq.H(alice), cirq.CNOT(alice,bob)])

    circuit.append([cirq.X(msg)**ranX, cirq.Y(msg)**ranY])

    circuit.append([cirq.CNOT(msg,alice), cirq.H(msg)])
    circuit.append(cirq.measure(msg,alice))

    circuit.append([cirq.CNOT(alice,bob), cirq.CZ(msg,bob)])

    return msg, circuit

def main():
    ranX = rn.random()
    ranY = rn.random()
    msg, circuit = qc_teleport(ranX, ranY)
    print(ranX, ranY)

    sim = cirq.Simulator()
    message = sim.simulate(cirq.Circuit([cirq.X(msg)**ranX, cirq.Y(msg)**ranY]))

    print('bloch sphere')
    b0x, b0y, b0z = cirq.bloch_vector_from_state_vector(message.final_state_vector, 0)
    print(round(b0x,2), round(b0y, 2), round(b0z, 2))
    print(circuit)
    final_result = sim.simulate(circuit)
    b2x, b2y, b2z = cirq.bloch_vector_from_state_vector(final_result.final_state_vector, 2)

    print(round(b2x,2), round(b2y, 2), round(b2z, 2))

if __name__ == '__main__':
    main()