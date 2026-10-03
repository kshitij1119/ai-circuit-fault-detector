"""Circuit fault definitions shared by simulation and training workflows."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Fault:
    code: str
    name: str
    description: str
    r1: float = 10_000.0
    r2: float = 10_000.0
    c1: float = 10e-9
    c2: float = 10e-9
    c2_short_r: float = 1e12
    opamp_gain: float = 100_000.0
    output_resistance: float = 100.0


FAULTS = (
    Fault("F0", "Healthy", "Nominal Sallen-Key band-pass circuit."),
    Fault("F1", "R1-open", "R1 is open; represented by a 1 TΩ leakage resistance.", r1=1e12),
    Fault("F2", "R2-short", "R2 is shorted; represented by a 1 mΩ resistance.", r2=1e-3),
    Fault("F3", "C1-open", "C1 is open; represented by a 1 fF leakage capacitance.", c1=1e-15),
    Fault("F4", "C2-short", "C2 is bypassed by a 1 mΩ short-circuit path.", c2_short_r=1e-3),
    Fault("F5", "Opamp-degraded", "Op-amp open-loop gain is reduced and output resistance is increased.", opamp_gain=20.0, output_resistance=50_000.0),
)

FAULT_BY_CODE = {fault.code: fault for fault in FAULTS}
FAULT_BY_NAME = {fault.name: fault for fault in FAULTS}
FAULT_BY_LABEL = {f"{fault.code}_{fault.name}": fault for fault in FAULTS}
FAULT_LABELS = tuple(FAULT_BY_LABEL)


def fault_label(code: str) -> str:
    try:
        fault = FAULT_BY_CODE[code]
    except KeyError as exc:
        raise ValueError(f"Unknown fault code {code!r}; choose from {', '.join(FAULT_BY_CODE)}") from exc
    return f"{fault.code}_{fault.name}"
