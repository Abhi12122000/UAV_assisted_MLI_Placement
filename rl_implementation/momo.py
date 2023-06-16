import gurobipy as gp
# Create a Gurobi environment
env = gp.Env()

# Set the username parameter
env.setParam('Username', 'shlokanegi')

# Solve a model
model = gp.Model()
model.optimize()