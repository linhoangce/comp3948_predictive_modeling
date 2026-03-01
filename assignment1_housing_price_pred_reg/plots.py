import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Load your processed data
df = pd.read_csv('data/df_new.csv')  # Your processed data with all features

# Features in your final model
final_features = [
    "accommodates", "bathrooms", "cleaning_fee", "host_identity_verified",
    "number_of_reviews", "review_scores_rating", "bedrooms", "beds",
    "neighbourhood_avg_price", "property_type_grouped_Other",
    "room_type_Private room", "room_type_Shared room",
    "cancellation_policy_strict", "cancellation_policy_super_strict_60",
    "city_NYC", "city_SF", "accommodates_squared", "bedrooms_squared",
    "beds_squared", "accommodates_bedrooms", "accommodates_bathrooms",
    "bedrooms_bathrooms", "accommodates_beds", "LA_neighbourhood_price",
    "SF_neighbourhood_price", "NYC_neighbourhood_price", "LA_accommodates",
    "SF_accommodates", "NYC_accommodates", "Chicago_accommodates",
    "private_room_bedrooms", "shared_room_accommodates", "house_bedrooms",
    "reviews_rating", "reviews_per_accommodates", "has_reviews",
    "cleaning_per_accommodates", "verified_response_rate", "luxury_indicator"
]

y = df['price']

# ============================================================================
# 1. CORRELATION ANALYSIS
# ============================================================================

print("=" * 100)
print("CORRELATION ANALYSIS")
print("=" * 100)

# Calculate correlations with target
correlations = {}
for feature in final_features:
    if feature in df.columns:
        corr, p_value = pearsonr(df[feature], y)
        correlations[feature] = corr

# Sort by absolute correlation
corr_df = pd.DataFrame({
    'Feature': list(correlations.keys()),
    'Correlation': list(correlations.values())
})
corr_df['Abs_Correlation'] = abs(corr_df['Correlation'])
corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)

print("\nTop 20 Features by Correlation with Price:")
print(corr_df.head(20).to_string(index=False))

# Correlation heatmap for top features
plt.figure(figsize=(12, 10))
top_20_features = corr_df.head(20)['Feature'].tolist()
corr_matrix = df[top_20_features + ['price']].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap: Top 20 Features vs Price', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 2. CORRELATION BAR PLOT
# ============================================================================

plt.figure(figsize=(14, 8))
colors = ['green' if x > 0 else 'red' for x in corr_df.head(20)['Correlation']]
plt.barh(range(len(corr_df.head(20))), corr_df.head(20)['Correlation'], color=colors, alpha=0.7)
plt.yticks(range(len(corr_df.head(20))), corr_df.head(20)['Feature'])
plt.xlabel('Correlation with Price', fontsize=12)
plt.title('Top 20 Feature Correlations with Price', fontsize=14, pad=20)
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('correlation_barplot.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 3. SCATTER PLOTS FOR KEY CONTINUOUS FEATURES
# ============================================================================

key_continuous_features = [
    'accommodates', 'bedrooms', 'bathrooms', 'neighbourhood_avg_price',
    'number_of_reviews', 'review_scores_rating', 'cleaning_fee', 'beds'
]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, feature in enumerate(key_continuous_features):
    if feature in df.columns:
        axes[idx].scatter(df[feature], y, alpha=0.3, s=10)
        axes[idx].set_xlabel(feature, fontsize=10)
        axes[idx].set_ylabel('Price', fontsize=10)
        axes[idx].set_title(f'{feature} vs Price\n(corr: {correlations.get(feature, 0):.3f})',
                            fontsize=11)

        # Add trend line
        z = np.polyfit(df[feature], y, 1)
        p = np.poly1d(z)
        axes[idx].plot(df[feature], p(df[feature]), "r--", alpha=0.8, linewidth=2)
        axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('scatter_plots_key_features.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 4. DISTRIBUTION ANALYSIS
# ============================================================================

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, feature in enumerate(key_continuous_features):
    if feature in df.columns:
        axes[idx].hist(df[feature], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[idx].set_xlabel(feature, fontsize=10)
        axes[idx].set_ylabel('Frequency', fontsize=10)
        axes[idx].set_title(f'Distribution of {feature}', fontsize=11)
        axes[idx].axvline(df[feature].mean(), color='red', linestyle='--',
                          linewidth=2, label=f'Mean: {df[feature].mean():.2f}')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('distribution_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 5. CATEGORICAL FEATURES ANALYSIS
# ============================================================================

categorical_features = [
    ('room_type_Private room', 'Private Room'),
    ('room_type_Shared room', 'Shared Room'),
    ('property_type_grouped_Other', 'Other Property Type'),
    ('cancellation_policy_strict', 'Strict Cancellation'),
    ('city_NYC', 'NYC'),
    ('city_SF', 'San Francisco')
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (feature, label) in enumerate(categorical_features):
    if feature in df.columns:
        box_data = [y[df[feature] == 0], y[df[feature] == 1]]
        bp = axes[idx].boxplot(box_data, labels=['No', 'Yes'], patch_artist=True)

        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')

        axes[idx].set_ylabel('Price', fontsize=10)
        axes[idx].set_xlabel(label, fontsize=10)
        axes[idx].set_title(f'Price by {label}\n(corr: {correlations.get(feature, 0):.3f})',
                            fontsize=11)
        axes[idx].grid(alpha=0.3, axis='y')

        # Add mean values
        mean_0 = y[df[feature] == 0].mean()
        mean_1 = y[df[feature] == 1].mean()
        axes[idx].text(0.5, 0.95, f'Mean (No): ${mean_0:.2f}\nMean (Yes): ${mean_1:.2f}',
                       transform=axes[idx].transAxes, fontsize=9,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('categorical_features_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 6. INTERACTION FEATURES ANALYSIS
# ============================================================================

interaction_features = [
    'accommodates_bedrooms', 'bedrooms_bathrooms', 'accommodates_bathrooms',
    'reviews_rating', 'luxury_indicator', 'NYC_neighbourhood_price'
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, feature in enumerate(interaction_features):
    if feature in df.columns:
        axes[idx].scatter(df[feature], y, alpha=0.3, s=10, color='purple')
        axes[idx].set_xlabel(feature, fontsize=10)
        axes[idx].set_ylabel('Price', fontsize=10)
        axes[idx].set_title(f'{feature} vs Price\n(corr: {correlations.get(feature, 0):.3f})',
                            fontsize=11)

        # Add trend line
        z = np.polyfit(df[feature], y, 1)
        p = np.poly1d(z)
        axes[idx].plot(df[feature], p(df[feature]), "r--", alpha=0.8, linewidth=2)
        axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('interaction_features_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 7. FEATURE IMPORTANCE FROM COEFFICIENTS
# ============================================================================

from production import COEFS

coef_df = pd.DataFrame({
    'Feature': list(COEFS.keys()),
    'Coefficient': list(COEFS.values())
})
coef_df = coef_df[coef_df['Feature'] != 'const']
coef_df['Abs_Coefficient'] = abs(coef_df['Coefficient'])
coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)

plt.figure(figsize=(14, 10))
colors = ['green' if x > 0 else 'red' for x in coef_df.head(20)['Coefficient']]
plt.barh(range(len(coef_df.head(20))), coef_df.head(20)['Coefficient'], color=colors, alpha=0.7)
plt.yticks(range(len(coef_df.head(20))), coef_df.head(20)['Feature'])
plt.xlabel('Coefficient Value (Standardized)', fontsize=12)
plt.title('Top 20 Feature Coefficients (Model Impact)', fontsize=14, pad=20)
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_coefficients.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 8. SUMMARY STATISTICS TABLE
# ============================================================================

print("\n" + "=" * 100)
print("FEATURE IMPACT SUMMARY")
print("=" * 100)

summary_df = pd.DataFrame({
    'Feature': coef_df.head(15)['Feature'],
    'Coefficient': coef_df.head(15)['Coefficient'],
    'Correlation': [correlations.get(f, 0) for f in coef_df.head(15)['Feature']]
})

print("\nTop 15 Features by Model Impact:")
print(summary_df.to_string(index=False))

# Save summary to CSV
summary_df.to_csv('feature_impact_summary.csv', index=False)
print("\nSummary saved to 'feature_impact_summary.csv'")

print("\n" + "=" * 100)
print("KEY INSIGHTS")
print("=" * 100)

print("\nPOSITIVE IMPACT (Increase Price):")
positive_features = coef_df[coef_df['Coefficient'] > 0].head(10)
for _, row in positive_features.iterrows():
    print(f"  • {row['Feature']}: +{row['Coefficient']:.4f}")

print("\nNEGATIVE IMPACT (Decrease Price):")
negative_features = coef_df[coef_df['Coefficient'] < 0].head(10)
for _, row in negative_features.iterrows():
    print(f"  • {row['Feature']}: {row['Coefficient']:.4f}")