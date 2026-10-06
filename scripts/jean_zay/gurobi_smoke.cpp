#include <cmath>
#include <iostream>
#include "gurobi_c++.h"

int main() {
  GRBEnv env(true);
  env.set(GRB_IntParam_OutputFlag, 0);
  env.start();
  GRBModel model(env);
  GRBVar x = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "x");
  model.setObjective(x, GRB_MAXIMIZE);
  model.optimize();
  if (model.get(GRB_IntAttr_Status) != GRB_OPTIMAL || std::abs(x.get(GRB_DoubleAttr_X) - 1.0) > 1e-9) return 1;
  std::cout << "Gurobi C++ API smoke test passed\n";
}
