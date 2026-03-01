import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

# Load your test predictions
# Assuming you have actual prices and predictions from your test set
# You'll need to modify this based on how you stored your test results

# Option 1: If you have them stored separately
# y_test = pd.read_csv('y_test.csv')['price']
# y_pred = pd.read_csv('y_pred.csv')['prediction']

# Option 2: Generate from your existing code

# Load and process test data
df = pd.read_csv('data/AirBNB1.csv')
y_pred = pd.read_csv('predictions.csv')
print(y_pred.max())

# Get actual prices (before processing removes them)
y_actual = df['price'].values
y_pred = y_pred.squeeze().values


# Calculate metrics
r2 = r2_score(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
mae = np.mean(np.abs(y_actual - y_pred))

# ============================================================================
# 1. ACTUAL VS PREDICTED SCATTER PLOT
# ============================================================================

plt.figure(figsize=(12, 10))
plt.scatter(y_actual, y_pred, alpha=0.4, s=20, color='blue', edgecolors='none')

# Perfect prediction line
max_val = max(y_actual.max(), y_pred.max())
min_val = min(y_actual.min(), y_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], 'g--', linewidth=2, label='Perfect Prediction')

# Add regression line
z = np.polyfit(y_actual, y_pred, 1)
p = np.poly1d(z)
plt.plot(y_actual, p(y_actual), 'r-', linewidth=2, alpha=0.8, label='Fitted Line')

plt.xlabel('Actual Price ($)', fontsize=14)
plt.ylabel('Predicted Price ($)', fontsize=14)
plt.title(f'Actual vs Predicted Prices\nR² = {r2:.4f}, RMSE = ${rmse:.2f}, MAE = ${mae:.2f}',
          fontsize=16, pad=20)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 2. RESIDUAL PLOT
# ============================================================================

residuals = y_actual - y_pred

plt.figure(figsize=(12, 6))
plt.scatter(y_pred, residuals, alpha=0.4, s=20, color='purple', edgecolors='none')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted Price ($)', fontsize=14)
plt.ylabel('Residuals (Actual - Predicted) ($)', fontsize=14)
plt.title('Residual Plot', fontsize=16, pad=20)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('residual_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 3. RESIDUAL DISTRIBUTION
# ============================================================================

plt.figure(figsize=(12, 6))
plt.hist(residuals, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
plt.axvline(x=residuals.mean(), color='g', linestyle='--', linewidth=2,
            label=f'Mean Error: ${residuals.mean():.2f}')
plt.xlabel('Residuals (Actual - Predicted) ($)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of Residuals', fontsize=16, pad=20)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('residual_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 4. ACTUAL VS PREDICTED BY PRICE RANGE
# ============================================================================

# Create price bins
price_bins = [0, 100, 200, 300, 500, 1000, 5000]
bin_labels = ['$0-100', '$100-200', '$200-300', '$300-500', '$500-1000', '$1000+']

df_results = pd.DataFrame({
    'actual': y_actual,
    'predicted': y_pred,
    'residual': residuals,
    'price_bin': pd.cut(y_actual, bins=price_bins, labels=bin_labels)
})

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, bin_label in enumerate(bin_labels):
    bin_data = df_results[df_results['price_bin'] == bin_label]

    if len(bin_data) > 0:
        axes[idx].scatter(bin_data['actual'], bin_data['predicted'],
                          alpha=0.5, s=30, color='blue', edgecolors='none')

        # Perfect prediction line
        min_val = bin_data['actual'].min()
        max_val = bin_data['actual'].max()
        axes[idx].plot([min_val, max_val], [min_val, max_val],
                       'r--', linewidth=2, label='Perfect')

        bin_r2 = r2_score(bin_data['actual'], bin_data['predicted'])
        bin_rmse = np.sqrt(mean_squared_error(bin_data['actual'], bin_data['predicted']))

        axes[idx].set_xlabel('Actual Price ($)', fontsize=11)
        axes[idx].set_ylabel('Predicted Price ($)', fontsize=11)
        axes[idx].set_title(f'{bin_label}\nR²={bin_r2:.3f}, RMSE=${bin_rmse:.2f}\n(n={len(bin_data)})',
                            fontsize=12)
        axes[idx].legend(fontsize=9)
        axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('actual_vs_predicted_by_price_range.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 5. PERFORMANCE METRICS BY PRICE RANGE
# ============================================================================

metrics_by_range = []
for bin_label in bin_labels:
    bin_data = df_results[df_results['price_bin'] == bin_label]
    if len(bin_data) > 0:
        metrics_by_range.append({
            'Price Range': bin_label,
            'Count': len(bin_data),
            'R²': r2_score(bin_data['actual'], bin_data['predicted']),
            'RMSE': np.sqrt(mean_squared_error(bin_data['actual'], bin_data['predicted'])),
            'MAE': np.mean(np.abs(bin_data['residual'])),
            'Mean Actual': bin_data['actual'].mean(),
            'Mean Predicted': bin_data['predicted'].mean()
        })

metrics_df = pd.DataFrame(metrics_by_range)

print("\n" + "=" * 100)
print("PERFORMANCE METRICS BY PRICE RANGE")
print("=" * 100)
print(metrics_df.to_string(index=False))

# Save metrics
metrics_df.to_csv('performance_by_price_range.csv', index=False)

# Plot metrics by price range
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].bar(metrics_df['Price Range'], metrics_df['R²'], color='skyblue', alpha=0.7, edgecolor='black')
axes[0].set_ylabel('R² Score', fontsize=12)
axes[0].set_title('R² by Price Range', fontsize=14)
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(alpha=0.3, axis='y')

axes[1].bar(metrics_df['Price Range'], metrics_df['RMSE'], color='salmon', alpha=0.7, edgecolor='black')
axes[1].set_ylabel('RMSE ($)', fontsize=12)
axes[1].set_title('RMSE by Price Range', fontsize=14)
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(alpha=0.3, axis='y')

axes[2].bar(metrics_df['Price Range'], metrics_df['MAE'], color='lightgreen', alpha=0.7, edgecolor='black')
axes[2].set_ylabel('MAE ($)', fontsize=12)
axes[2].set_title('MAE by Price Range', fontsize=14)
axes[2].tick_params(axis='x', rotation=45)
axes[2].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('metrics_by_price_range.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 6. ERROR PERCENTAGE ANALYSIS
# ============================================================================

df_results['error_pct'] = (np.abs(residuals) / y_actual) * 100

plt.figure(figsize=(12, 6))
plt.hist(df_results['error_pct'], bins=100, alpha=0.7, color='coral', edgecolor='black')
plt.axvline(x=df_results['error_pct'].median(), color='r', linestyle='--',
            linewidth=2, label=f'Median: {df_results["error_pct"].median():.1f}%')
plt.xlabel('Absolute Percentage Error (%)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of Prediction Error Percentage', fontsize=16, pad=20)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.xlim(0, 100)
plt.tight_layout()
plt.savefig('error_percentage_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nMedian Absolute Percentage Error: {df_results['error_pct'].median():.2f}%")
print(f"Predictions within 20% error: {(df_results['error_pct'] <= 20).sum() / len(df_results) * 100:.2f}%")
print(f"Predictions within 30% error: {(df_results['error_pct'] <= 30).sum() / len(df_results) * 100:.2f}%")