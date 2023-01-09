import functions as f

# Generate ground truth polynomial.
w_gt = f.gen_ground_truth_model()

# Generate data.
stdev = 3
x, y = f.gen_data(w_gt, n=1000, stdev=stdev)

# Find the best fit model.
w = f.find_best_fit(x, y, reg=1e-3)

print(f'Ground Truth Model Dimensions: {w_gt.shape[0]}')
print(f'Best Fit Model Dimensions: {w.shape[0]}')

# Generate test points.
x_point, y_point = f.gen_data(w_gt, n=20, stdev=stdev)

# Plot the data and the best fit model.
f.plot_data(x, y, w, w_gt, stdev, x_point, y_point)