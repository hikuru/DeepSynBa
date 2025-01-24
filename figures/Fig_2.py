import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd


# Path to the uploaded file
file_path = 'case_study.json'

# Load the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)


### Case Study 1
cell_name = "786-0"
drug1 = "Bleomycin sulfate"
drug2 = "Cabazitaxel"
key_to_search = drug1 + "_" + drug2 + "_" + cell_name

# Extract the required fields
required_fields = ["e1_mean", "e2_mean", "e3_mean", "logC1_mean", "logC2_mean", "h1_mean", "h2_mean", "alpha_mean"]
extracted_values = {field: data[key_to_search].get(field, None) for field in required_fields}

dose_response = None
dose_response_gt = None
for key, value in data.items():
    if value.get('drug2') == drug2:
        dose_response = value.get('dose_response', [])
        dose_response_gt = value.get('dose_response_gt', [])
        break

# Convert the dose_response and dose_response_gt into 4x4 matrices (if they exist)
dose_response_matrix = np.array(dose_response) if dose_response else pd.DataFrame()
dose_response_gt_matrix = np.array(dose_response_gt) if dose_response_gt else pd.DataFrame()

# Display the predicted and the true dose-response matrices
print(dose_response_matrix, dose_response_gt_matrix)

# Parameters
e_0 = 100
e_1 = extracted_values.get('e1_mean')
e_2 = extracted_values.get('e2_mean')
e_3 = extracted_values.get('e3_mean')
logC_1 = extracted_values.get('logC1_mean')
logC_2 = extracted_values.get('logC2_mean')
h_1 = extracted_values.get('h1_mean')
h_2 = extracted_values.get('h2_mean')
alpha = extracted_values.get('alpha_mean')
sigma = 1.0       # standard deviation for noise
add_noise = True  # toggle for adding noise

# Dosages
drug1_dose = data[key_to_search].get("drug1_dose", [])
drug2_dose = data[key_to_search].get("drug2_dose", [])
X1_points, X2_points = np.meshgrid(drug1_dose, drug2_dose)

# Create a grid of x1 and x2 values
x1 = np.logspace(-4, np.ceil(np.log10(np.max(drug1_dose)) + 2), 5000)  # Log-spaced values for x1
x2 = np.logspace(-4, np.ceil(np.log10(np.max(drug2_dose)) + 2), 5000)  # Log-spaced values for x2

X1, X2 = np.meshgrid(x1, x2)

# Calculate intermediate variables
def synba_2d(X1, X2):
    A = (np.exp(logC_1) ** h_1) * (np.exp(logC_2) ** h_2) * e_0
    B = (X1 ** h_1) * (np.exp(logC_2) ** h_2) * e_1 * e_0
    C = (X2 ** h_2) * (np.exp(logC_1) ** h_1) * e_2 * e_0
    D = alpha * (X1 ** h_1) * (X2 ** h_2) * e_3 * e_0
    AA = (np.exp(logC_1) ** h_1) * (np.exp(logC_2) ** h_2)
    BB = (X1 ** h_1) * (np.exp(logC_2) ** h_2)
    CC = (X2 ** h_2) * (np.exp(logC_1) ** h_1)
    DD = alpha * (X1 ** h_1) * (X2 ** h_2)
    Y = (A + B + C + D) / (AA + BB + CC + DD)
    return Y

# Plot the slices for Drug 1
plt.figure(figsize=(5, 5))
for i in range(4):
    x1_fixed = drug1_dose[i]
    print(x1_fixed)
    y = synba_2d(x1_fixed, x2)
    formatted_dose = f'{x1_fixed:.1g}' if x1_fixed < 1 else f'{int(round(x1_fixed))}'
    plt.scatter(np.log(np.maximum(np.min(x2), np.array(drug2_dose))), dose_response_gt_matrix[i, :], label=f'$x_1$ = {formatted_dose}')
    plt.plot(np.log(x2), y, label=None)
plt.legend(loc='upper right', frameon=True, borderaxespad=0.1, fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(str(key_to_search) + '_drug1.pdf')  # png
plt.show()

# Plot the slices for Drug 2
plt.figure(figsize=(5, 5))
for i in range(4):
    x2_fixed = drug2_dose[i]
    print(x2_fixed)
    y = synba_2d(x1, x2_fixed)
    formatted_dose = f'{x2_fixed:.1g}' if x2_fixed < 1 else f'{int(round(x2_fixed))}'
    plt.scatter(np.log(np.maximum(np.min(x1), np.array(drug1_dose))), dose_response_gt_matrix[:, i], label=f'$x_2$ = {formatted_dose}')
    plt.plot(np.log(x1), y, label=None)
plt.legend(loc='upper right', frameon=True, borderaxespad=0.1, fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(str(key_to_search) + '_drug2.pdf')  # png
plt.show()


# Plot the surface
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([2, 2, 2])
Y = synba_2d(X1, X2)
surf = ax.plot_surface(np.log(X1), np.log(X2), Y, cmap='viridis', edgecolor='none', alpha=0.8)

# Plot the ground truth dose-response points
dose_response_gt = np.array(data[key_to_search].get("dose_response_gt", []))
ax.scatter(np.log(X1_points+1e-4), np.log(X2_points+1e-4), np.transpose(dose_response_gt), color='red', s=50, label='Ground truth dose points', zorder=10)

# Add color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Set the initial view angle (elevation, azimuth)
ax.view_init(elev=30, azim=45)  # 30 degrees up, 45 degrees to the right

plt.savefig(str(key_to_search) + '_surface_plot.pdf')
plt.show()

# Plot the contour plot
plt.figure(figsize=(5, 5))
contour = plt.contour(np.log(X1), np.log(X2), Y, levels=8, cmap='viridis')
plt.xlim(-10, 5)
plt.ylim(-9, -4)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.clabel(contour, inline=True, fontsize=10, fmt="%.2f")

plt.savefig(str(key_to_search) + '_contour_plot.pdf')
plt.show()

### End of Case Study 1


### Case Study 2
cell_name = "MDA-MB-468"
drug1 = "Azacitidine"
drug2 = "Triethylenemelamine"
key_to_search = drug1 + "_" + drug2 + "_" + cell_name

# Extract the required fields
required_fields = ["e1_mean", "e2_mean", "e3_mean", "logC1_mean", "logC2_mean", "h1_mean", "h2_mean", "alpha_mean"]
extracted_values = {field: data[key_to_search].get(field, None) for field in required_fields}

dose_response = None
dose_response_gt = None
for key, value in data.items():
    if value.get('drug2') == drug2:
        dose_response = value.get('dose_response', [])
        dose_response_gt = value.get('dose_response_gt', [])
        break

# Convert the dose_response and dose_response_gt into 4x4 matrices (if they exist)
dose_response_matrix = np.array(dose_response) if dose_response else pd.DataFrame()
dose_response_gt_matrix = np.array(dose_response_gt) if dose_response_gt else pd.DataFrame()

# Display the predicted and the true dose-response matrices
print(dose_response_matrix, dose_response_gt_matrix)

# Parameters
e_0 = 100
e_1 = extracted_values.get('e1_mean')
e_2 = extracted_values.get('e2_mean')
e_3 = extracted_values.get('e3_mean')
logC_1 = extracted_values.get('logC1_mean')
logC_2 = extracted_values.get('logC2_mean')
h_1 = extracted_values.get('h1_mean')
h_2 = extracted_values.get('h2_mean')
alpha = extracted_values.get('alpha_mean')
sigma = 1.0       # standard deviation for noise
add_noise = True  # toggle for adding noise

# Dosages
drug1_dose = data[key_to_search].get("drug1_dose", [])
drug2_dose = data[key_to_search].get("drug2_dose", [])
X1_points, X2_points = np.meshgrid(drug1_dose, drug2_dose)

# Create a grid of x1 and x2 values
x1 = np.logspace(-4, np.ceil(np.log10(np.max(drug1_dose)) + 2), 5000)  # Log-spaced values for x1
x2 = np.logspace(-4, np.ceil(np.log10(np.max(drug2_dose)) + 2), 5000)  # Log-spaced values for x2

X1, X2 = np.meshgrid(x1, x2)

# Calculate intermediate variables
def synba_2d(X1, X2):
    A = (np.exp(logC_1) ** h_1) * (np.exp(logC_2) ** h_2) * e_0
    B = (X1 ** h_1) * (np.exp(logC_2) ** h_2) * e_1 * e_0
    C = (X2 ** h_2) * (np.exp(logC_1) ** h_1) * e_2 * e_0
    D = alpha * (X1 ** h_1) * (X2 ** h_2) * e_3 * e_0
    AA = (np.exp(logC_1) ** h_1) * (np.exp(logC_2) ** h_2)
    BB = (X1 ** h_1) * (np.exp(logC_2) ** h_2)
    CC = (X2 ** h_2) * (np.exp(logC_1) ** h_1)
    DD = alpha * (X1 ** h_1) * (X2 ** h_2)
    Y = (A + B + C + D) / (AA + BB + CC + DD)
    return Y

# Plot the slices for Drug 1
plt.figure(figsize=(5, 5))
for i in range(4):
    x1_fixed = drug1_dose[i]
    print(x1_fixed)
    y = synba_2d(x1_fixed, x2)
    formatted_dose = f'{x1_fixed:.1g}' if x1_fixed < 1 else f'{int(round(x1_fixed))}'
    plt.scatter(np.log(np.maximum(np.min(x2), np.array(drug2_dose))), dose_response_gt_matrix[i, :], label=f'$x_1$ = {formatted_dose}')
    plt.plot(np.log(x2), y, label=None)
plt.legend(loc='upper right', frameon=True, borderaxespad=0.1, fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(str(key_to_search) + '_drug1.pdf')  # png
plt.show()

# Plot the slices for Drug 2
plt.figure(figsize=(5, 5))
for i in range(4):
    x2_fixed = drug2_dose[i]
    print(x2_fixed)
    y = synba_2d(x1, x2_fixed)
    formatted_dose = f'{x2_fixed:.1g}' if x2_fixed < 1 else f'{int(round(x2_fixed))}'
    plt.scatter(np.log(np.maximum(np.min(x1), np.array(drug1_dose))), dose_response_gt_matrix[:, i], label=f'$x_2$ = {formatted_dose}')
    plt.plot(np.log(x1), y, label=None)
plt.legend(loc='upper right', frameon=True, borderaxespad=0.1, fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(str(key_to_search) + '_drug2.pdf')  # png
plt.show()


# Plot the surface
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([2, 2, 2])
Y = synba_2d(X1, X2)
surf = ax.plot_surface(np.log(X1), np.log(X2), Y, cmap='viridis', edgecolor='none', alpha=0.8)

# Plot the ground truth dose-response points
dose_response_gt = np.array(data[key_to_search].get("dose_response_gt", []))
ax.scatter(np.log(X1_points+1e-4), np.log(X2_points+1e-4), np.transpose(dose_response_gt), color='red', s=50, label='Ground truth dose points', zorder=10)

# Add color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Set the initial view angle (elevation, azimuth)
ax.view_init(elev=30, azim=45)  # 30 degrees up, 45 degrees to the right

plt.savefig(str(key_to_search) + '_surface_plot.pdf')
plt.show()

# Plot the contour plot
plt.figure(figsize=(5, 5))
contour = plt.contour(np.log(X1), np.log(X2), Y, levels=10, cmap='viridis')
plt.xlim(-4, 4)
plt.ylim(-4, 6)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.clabel(contour, inline=True, fontsize=10, fmt="%.2f")

plt.savefig(str(key_to_search) + '_contour_plot.pdf')
plt.show()

