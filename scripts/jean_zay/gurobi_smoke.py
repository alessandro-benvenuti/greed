import gurobipy as gp

model = gp.Model("greed_license_smoke")
x = model.addVar(lb=0.0, name="x")
model.setObjective(x, gp.GRB.MAXIMIZE)
model.addConstr(x <= 1.0)
model.optimize()
assert model.Status == gp.GRB.OPTIMAL and abs(x.X - 1.0) < 1e-9
print(f"Gurobi {gp.gurobi.version()} compute-node licence smoke test passed")
