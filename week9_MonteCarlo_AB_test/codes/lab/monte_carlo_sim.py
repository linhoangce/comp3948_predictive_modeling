import numpy as np
import pandas as pd
from scipy.stats import norm

NUM_SIM = 500
REV_EXPECTED = 194000
REV_SD = 15000

FC_EXPECTED = 60000
FC_SD = 4000
VC_EXPECTED = 100000
VC_SD = 40000

def generate_random_numbers(mean, sd):
    return norm.rvs(loc=mean, scale=sd, size=NUM_SIM)

revenues = generate_random_numbers(mean=REV_EXPECTED, sd=REV_SD)
fixed_costs = generate_random_numbers(mean=FC_EXPECTED, sd=FC_SD)
variable_costs = generate_random_numbers(mean=VC_EXPECTED, sd=VC_SD)


df = pd.DataFrame()
profits = [revenues[i] - fixed_costs[i] - variable_costs[i] for i in range(len(revenues))]
df['Revenue'] = np.round(revenues, 2)
df['Fixed Cost'] = np.round(fixed_costs, 2)
df['Variable Cost'] = np.round(variable_costs, 2)
df['Profit'] = profits

print(df)

print(f"Profit mean: {df['Profit'].mean()}")
print(f"Profit SD: {df['Profit'].std()}")
print(f"Profit Min: {df['Profit'].min()}")
print(f"Profit Max: {df['Profit'].max()}")

# Calculate the risk of incurring a loss
df_loss = df[(df['Profit'] < 0)]
total_loss = df_loss['Profit'].count()
risk_of_loss = total_loss / NUM_SIM
print(f"Risk of loss: {risk_of_loss}")