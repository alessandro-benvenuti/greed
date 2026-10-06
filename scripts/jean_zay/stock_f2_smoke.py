import pyged

lower, upper = pyged.sed(([0], []), ([0], []), ["ged_f2"], ["--time-limit 10 --threads 1 --map-root-to-root FALSE"])
assert lower == upper == 0.0, (lower, upper)
print(f"GEDLIB stock F2 smoke test passed with Gurobi {pyged.solver_version()}")
