#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# In[9]:


get_ipython().system('pip install numpy-financial')


# In[11]:


import numpy as np
import numpy_financial as npf


# In[15]:


# Load the dataset

file_path = r'C:\Users\HP\Downloads\lagos_rooftop_solar_potential.csv'
df = pd.read_csv(file_path)
df.head()


# In[3]:


# Clean data

df_clean = df.drop(columns=['uuid', 'City', 'Comment', 'Unit_installation_price'], errors='ignore').dropna()
le = LabelEncoder()
df_clean['Assumed_building_type'] = le.fit_transform(df_clean['Assumed_building_type'])


# In[4]:


# Economic Assumptions

cost_per_kw = 550000  # NGN
price_per_kwh = 75    # NGN
lifetime = 25
discount = 0.10
maintenance_rate = 0.01
carbon_price_usd = 30


# In[5]:


# Top 10 energy potential buildings

top10 = df_clean.sort_values('Energy_potential_per_year', ascending=False).head(10).copy()
top10['Investment_NGN'] = top10['Peak_installable_capacity'] * cost_per_kw
top10['Annual_savings_NGN'] = top10['Energy_potential_per_year'] * price_per_kwh
top10['Annual_maintenance'] = top10['Investment_NGN'] * maintenance_rate
top10['Net_annual_cashflow'] = top10['Annual_savings_NGN'] - top10['Annual_maintenance']


# In[6]:


# NPV Function

def npv(cashflow, rate, years):
    return sum([cashflow / ((1 + rate) ** y) for y in range(1, years + 1)])
    


# In[12]:


# IRR Function
def irr(initial, cashflow, years):
    cashflows = [-initial] + [cashflow] * years
    return npf.irr(cashflows)
    


# In[13]:


# Calculations

top10['NPV_NGN'] = top10['Net_annual_cashflow'].apply(lambda cf: npv(cf, discount, lifetime))
top10['IRR'] = top10.apply(lambda row: irr(row['Investment_NGN'], row['Net_annual_cashflow'], lifetime), axis=1)
top10['Payback_period'] = top10['Investment_NGN'] / top10['Net_annual_cashflow']
top10['Lifetime_energy_kwh'] = top10['Energy_potential_per_year'] * lifetime
top10['Total_cost'] = top10['Investment_NGN'] + (top10['Annual_maintenance'] * lifetime)
top10['LCOE_NGN_per_kwh'] = top10['Total_cost'] / top10['Lifetime_energy_kwh']
top10['CO2_offset_tons'] = top10['Lifetime_energy_kwh'] * 0.0007
top10['Carbon_credit_value_USD'] = top10['CO2_offset_tons'] * carbon_price_usd


# In[14]:


top10_final = top10[[
    'Peak_installable_capacity', 'Energy_potential_per_year', 'Investment_NGN',
    'Annual_savings_NGN', 'NPV_NGN', 'IRR', 'Payback_period',
    'LCOE_NGN_per_kwh', 'CO2_offset_tons', 'Carbon_credit_value_USD'
]].round(2)

top10_final


# # 1. Net Present Value (NPV)
# All top 10 projects have strong positive NPVs, ranging from: ₦1.84 billion to ₦4.06 billion.
# This means each rooftop solar investment will generate more value over 25 years than it costs to install — even after accounting for time and discounting.
# * Best project: Row 0 (Project ID 108183) stands out with ₦4.06 billion NPV, the highest by far.
# 
# # 2. Payback Period
# Payback periods range from 6.00 to 6.26 years.
# Meaning: The initial solar investment is recovered in just over 6 years from electricity savings.
# * Quickest ROI: Projects with lowest payback (~6.00 years) are great for short-term cost recovery.
# 
# # 3. IRR (Internal Rate of Return)
# IRR is consistently 16%, indicating robust long-term profitability.
# For comparison, IRR >10% is typically considered excellent in infrastructure/energy sectors.
# 
# # 4. Levelized Cost of Energy (LCOE)
# LCOE is in the range of ₦21.2 – ₦22.1/kWh.
# In Lagos, this is significantly lower than grid/diesel backup rates, which often exceed ₦70–₦100/kWh.
# 
# # 5. CO₂ Offset and Carbon Credit Revenue
# Projects offset between ~50,000 to 111,000 tons of CO₂ over 25 years.
# Monetized via global carbon markets (at $30/ton), each project could yield: $1.5M to $3.3M in carbon credits.
# 
# 

# In[23]:


import matplotlib.pyplot as plt


# In[16]:


# Electrification Gap Analysis with simulated data 

# Simulated data: 10 LGAs in Lagos
lga_names = [
    "Ikeja", "Eti-Osa", "Ikorodu", "Badagry", "Alimosho",
    "Surulere", "Epe", "Lagos Island", "Mushin", "Oshodi-Isolo"
]


# In[17]:


# Electrification access (0 = poor, 1 = full)

np.random.seed(42)
electrification_rate = np.random.uniform(0.3, 0.9, size=10)


# In[18]:


# Simulated solar potential (MW)

solar_capacity_mw = np.random.uniform(50, 300, size=10)


# In[19]:


# Combine into DataFrame

lga_df = pd.DataFrame({
    "LGA": lga_names,
    "Electrification_rate": electrification_rate,
    "Solar_capacity_MW": solar_capacity_mw
})


# In[20]:


# Compute priority: low electrification + high solar = high investment score

lga_df["Investment_priority_score"] = (1 - lga_df["Electrification_rate"]) * lga_df["Solar_capacity_MW"]


# In[21]:


# Sort for visualisation

lga_df_sorted = lga_df.sort_values("Investment_priority_score", ascending=False)


# In[24]:


# Plot

plt.figure(figsize=(10, 6))
plt.barh(lga_df_sorted["LGA"], lga_df_sorted["Investment_priority_score"], color='darkgreen')
plt.xlabel("Investment Priority Score (Low Elec. × High Solar)")
plt.title("Electrification Gap vs Solar Potential: Investment Targeting")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# In[25]:


# View top 5 LGAs to target

lga_df_sorted.head()


# # Binomial Tree Model – Real Options Analysis
# 
# *  Goal: Evaluate the option to delay solar investment for 1–3 years, based on possible cost reductions and energy price fluctuations.
# *  Real Options Idea:
#                     I don’t have to invest now — I (investor) can wait and decide later if conditions improve. The binomial tree models this                                uncertainty using "up" and "down" moves.
# 

# In[26]:


# Defining my parameters

initial_cost = 550000  # NGN per kW (starting point)
expected_savings = 600000  # Annual savings per kW if installed
years = 3  # Decision horizon
discount_rate = 0.10  # 10% time value of money
volatility = 0.15  # Assumed volatility in installation cost (15%)


# In[27]:


# Compute up and down factors

u = 1 + volatility       # Upward cost movement
d = 1 - volatility       # Downward cost movement
p = 0.5                  # Assume equal probabilities for simplicity


# In[28]:


# Build the binomial tree of installation costs

tree = [[0] * (i + 1) for i in range(years + 1)]

for i in range(years + 1):
    for j in range(i + 1):
        tree[i][j] = initial_cost * (u ** j) * (d ** (i - j))
        


# In[29]:


# Evaluate project value at final year

project_value = [[0] * (i + 1) for i in range(years + 1)]
for j in range(years + 1):
    net_cashflow = expected_savings - tree[years][j]
    project_value[years][j] = max(net_cashflow / discount_rate, 0)  # Only invest if positive NPV


# In[30]:


# Work backwards through tree (dynamic programming)

for i in reversed(range(years)):
    for j in range(i + 1):
        # Discounted expected value of continuing (not investing yet)
        continuation = (p * project_value[i+1][j+1] + (1 - p) * project_value[i+1][j]) / (1 + discount_rate)

        # Immediate exercise value (if you invest now)
        exercise = max(expected_savings - tree[i][j], 0) / discount_rate

        # Choose the higher value (option to wait vs act now)
        project_value[i][j] = max(continuation, exercise)
        


# In[31]:


# Final result

print(f" Real Option Value of Waiting: ₦{project_value[0][0]:,.2f}")


# # The Real Option Value of ₦736,587.86 means there is some benefit to waiting, but not a huge one — possibly because:
# * The expected savings aren't far from the installation cost.
# * Volatility isn’t high enough to make delay highly valuable.

# # Stochastic Monte Carlo Simulation – Advanced Risk Analysis
# * Goal: Simulate thousands of possible futures to assess risk and expected return of a solar project under uncertainty (such as costs, energy output, inflation).
# # Key Uncertainties:
# * Installation cost (₦/kW)
# * Annual energy output (kWh)
# * Electricity price (₦/kWh)
# * System lifespan (years)
# * Discount rate

# In[32]:


# Define base values

base_install_cost = 550000        # NGN per kW
base_energy_output = 115000       # kWh per year
base_energy_price = 75            # NGN per kWh
lifespan = 25                     # Years
discount_rate = 0.10              # Time value of money


# In[33]:


# Define simulation size and empty results list

simulations = 10000
npvs = []


# In[37]:


# Run simulations

for _ in range(simulations):
    # Randomly vary inputs using normal or lognormal distributions
    install_cost = np.random.normal(loc=base_install_cost, scale=30000)  # ~±5.5%
    energy_output = np.random.normal(loc=base_energy_output, scale=8000) # ~±7%
    energy_price = np.random.normal(loc=base_energy_price, scale=5)      # ~±7%

    # Ensure all values are realistic (no negatives)
    install_cost = max(install_cost, 400000)
    energy_output = max(energy_output, 80000)
    energy_price = max(energy_price, 50)

    # Calculate annual savings
    annual_cashflow = energy_output * energy_price
    maintenance = install_cost * 0.01
    net_cashflow = annual_cashflow - maintenance

    # Compute NPV over project lifespan
    npv = sum([net_cashflow / ((1 + discount_rate) ** y) for y in range(1, lifespan + 1)])
    npv -= install_cost

    npvs.append(npv)
    


# In[38]:


# Plot distribution of NPVs

plt.figure(figsize=(10, 5))
plt.hist(npvs, bins=50, color='steelblue', edgecolor='black')
plt.axvline(np.mean(npvs), color='red', linestyle='--', label=f"Mean NPV = ₦{np.mean(npvs):,.0f}")
plt.title("Monte Carlo Simulation: Project NPV Distribution")
plt.xlabel("Net Present Value (NGN)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()


# In[39]:


# Step 6: Print results

print(f" Mean NPV: ₦{np.mean(npvs):,.2f}")
print(f" 10th percentile (worst-case): ₦{np.percentile(npvs, 10):,.2f}")
print(f" 90th percentile (best-case): ₦{np.percentile(npvs, 90):,.2f}")


# # I ran 10,000 risk simulations on a solar rooftop investment, accounting for uncertain factors like installation cost, energy output, and electricity prices. I can infer that:
# Mean NPV: ₦77,768,243.49
# * This is the average expected profit (after discounting and deducting installation cost). 
# * It represents the best estimate of the net value created by the project over 25 years.
# # Deduction:
# This is a strong positive NPV, indicating the project is financially attractive under average market conditions.
# 
# # 10th Percentile (Worst-Case): ₦68,033,514.62
# * 90% of simulated outcomes are better than this value. Even in bad market conditions (low energy prices, high costs, low output), I can still earn ~₦68M.
# # Deduction:
# This project has low downside risk — it remains profitable even in adverse conditions.
# 
# # 90th Percentile (Best-Case): ₦87,527,280.09
# * Only 10% of simulations did better than this. This is the upside ceiling, showing potential for near ₦90M profit under favorable conditions.
# # Deduction:
# If installation costs fall or energy prices rise, the returns could be exceptional.
# 

# # Conclusively:
# * The investment is robust, low-risk, and profitable.
# * The narrow spread between worst- and best-case scenarios shows it's a predictable, stable project — ideal for climate-aligned infrastructure portfolios or green bonds.

# # Solar Portfolio-Level Monte Carlo Simulation
# + Carbon Credit Sensitivity Plot
# + Break-Even Carbon Price Estimation

# In[40]:


# Simulation settings

num_projects = 5
simulations = 10000
lifespan = 25
discount_rate = 0.10
carbon_prices = np.linspace(10, 100, 10)  # From $10 to $100 per ton


# In[41]:


# Simulate 5 rooftop projects with varying capacity

np.random.seed(42)
project_data = [{'name': f'Project_{i+1}', 'capacity_kw': np.random.randint(2000, 6000)} for i in range(num_projects)]

# Store carbon price vs NPV
results = []


# In[42]:


for carbon_price in carbon_prices:
    total_portfolio_npvs = []

    for project in project_data:
        capacity = project['capacity_kw']
        project_npvs = []

        for _ in range(simulations):
            # Simulate input variables
            install_cost = np.random.normal(550000, 30000) * capacity
            energy_output = np.random.normal(115000, 8000) * (capacity / 5000)
            energy_price = np.random.normal(75, 5)
            carbon_factor = 0.0007  # tCO2 per kWh

            # Adjustments for realism
            install_cost = max(install_cost, 400000 * capacity)
            energy_output = max(energy_output, 80000 * (capacity / 5000))
            energy_price = max(energy_price, 50)

            # Cashflows
            annual_savings = energy_output * energy_price
            maintenance = install_cost * 0.01
            net_cashflow = annual_savings - maintenance

            # Carbon credit income (converted to NGN at ₦1500/USD)
            co2_offset = energy_output * carbon_factor * lifespan
            carbon_revenue_ngn = co2_offset * carbon_price * 1500

            # Discounted cashflow (NPV)
            npv_energy = sum([net_cashflow / ((1 + discount_rate) ** y) for y in range(1, lifespan + 1)])
            npv_total = npv_energy + carbon_revenue_ngn - install_cost

            project_npvs.append(npv_total)

        # Mean NPV for one project
        total_portfolio_npvs.append(np.mean(project_npvs))

    # Total NPV across all 5 rooftops
    results.append({
        'Carbon_price_USD': carbon_price,
        'Portfolio_NPV_NGN': sum(total_portfolio_npvs)
    })


# In[43]:


# Convert to DataFrame

results_df = pd.DataFrame(results)


# In[44]:


#  Plot results

plt.figure(figsize=(10, 6))
plt.plot(results_df['Carbon_price_USD'], results_df['Portfolio_NPV_NGN'] / 1e9, marker='o', linestyle='-', color='green')
plt.axhline(0, color='red', linestyle='--')
plt.title('Carbon Credit Price vs Portfolio NPV (₦ Billion)')
plt.xlabel('Carbon Credit Price (USD/ton CO₂)')
plt.ylabel('Total Portfolio NPV (₦ Billion)')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[46]:


# Display data

print(" Carbon Price Sensitivity Table (Top 10 values):")
results_df.round(2)


# # Carbon Market Sensitivity Analysis
# I simulated how different carbon credit prices ($10–$100/ton CO₂) affect the Net Present Value (NPV) of the solar rooftop portfolio (5 projects).
# # Table Recap:
# | Carbon Price (USD/tCO₂) | Portfolio NPV (₦) | Change from \$10 |
# | ----------------------- | ----------------- | ---------------- |
# |  $10                    | -₦11.54 Billion   | Baseline         |
# |  $50                    | -₦11.07 Billion   | +₦0.47B          |
# |  $100                   | -₦10.46 Billion   | +₦1.08B          |
# 
# # Graph Summary:
# * The green line shows that NPV improves linearly as carbon price rises.
# * The red dashed line marks NPV = 0 — the breakeven point.
# * The portfolio remains in the red (losses) even at $100/tCO₂.

# # INFERENCE:
# 1. High Cost Structure
# Even with generous carbon credit pricing, the portfolio NPV remains strongly negative.
# Implies that installation, maintenance, or energy price assumptions are too conservative or outdated.
# 
# 2. Carbon Revenue Alone Is Not Enough
# Carbon credit income adds ₦1B+ at $100/ton, but the total shortfall is ₦11.5B.
# Carbon markets cannot close the viability gap alone — additional interventions are needed.
# 
# 3. Required Carbon Price for Breakeven
# Using linear interpolation:
# The NPV improves ~₦108M for each $10 increase in carbon price.
# To recover ₦11.5B:
# 11.5 B NGN            = ≈1064 USD/tCO₂
# 0.108 B NGN per $10
# 
# # The Breakeven carbon price ≈ $1,060 per ton CO₂ — not realistic in today's market. 
