import matplotlib.pyplot as plt
import pandas as pd
import textwrap
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv('data/df_new.csv')

y = df['class']

features = [ 'credit_amount', 'employment', 'checking_status_<0', 'age_bin_35-50', 'checking_status_no checking',
             'property_magnitude_no known property', 'purpose_new car','savings_status_<100',
             'credit_history_critical/other existing credit',
             'savings_status_no known savings', 'housing_rent', 'credit_history_no credits/all paid',
            ]



plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# X = df[features]

for i, feature in enumerate(['credit_amount', 'employment']):
    plt.subplot(1, 2, i+1)
    plt.hist(df[feature])
    plt.title(textwrap.fill(feature, width=20), fontsize=24)


plt.subplots_adjust(wspace=0.7, hspace=0.4)

plt.show()


plt.subplots(nrows=2, ncols=5, figsize=(24, 16))

# X = df[features]
features = ['checking_status_<0', 'age_bin_35-50', 'checking_status_no checking',
             'property_magnitude_no known property', 'purpose_new car','savings_status_<100',
             'credit_history_critical/other existing credit',
             'savings_status_no known savings', 'housing_rent', 'credit_history_no credits/all paid',]
for i, feature in enumerate(features):
    plt.subplot(2, 5, i+1)
    plt.hist(df[feature])
    plt.title(textwrap.fill(feature, width=20), fontsize=24)


plt.subplots_adjust(wspace=0.7, hspace=0.4)

plt.show()

plt.figure(figsize=(24, 16))

corr = df[['class']+features].corr()

sns.heatmap(corr,
            annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            annot_kws={"size": 18})
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout(pad=0)
plt.show()

plt.figure(figsize=(25, 16))
corr = corr.sort_values(by=['class'], ascending=False)

sns.heatmap(corr[['class']], annot=True, linewidths=0.1,
            vmin=-0.5, vmax=1, cmap='YlGnBu', annot_kws={"size": 18})
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.show()


def compute_vif(df, features):
    X = df[features]
    vif_data = []

    for i, col in enumerate(X.columns):
        vif = variance_inflation_factor(X.values, i)
        vif_data.append([col, vif])

    vif_df = pd.DataFrame(vif_data, columns=['Features', 'VIF'])
    return vif_df.sort_values(by='VIF', ascending=True)

vif_table = compute_vif(df, features)
print(vif_table)


plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='class', y='credit_amount', )
plt.title('Credit Amount by Class')
plt.xlabel('Class')
plt.ylabel('Credit Amount')
plt.tight_layout()
plt.show()

binary_features = ['checking_status_<0', 'age_bin_35-50', 'checking_status_no checking',
             'property_magnitude_no known property', 'purpose_new car','savings_status_<100',
             'credit_history_critical/other existing credit',
             'savings_status_no known savings', 'housing_rent', 'credit_history_no credits/all paid']

fig, axes = plt.subplots(2, 2, figsize=(14, 16))

for i, feature in enumerate(binary_features[:4]):
    rate = df.groupby(feature)['class'].mean()

    ax = axes[i // 2, i % 2]  # Select subplot

    sns.barplot(x=rate.index.astype(str), y=rate.values, ax=ax)

    # Wrap long title
    wrapped_title = textwrap.fill(f'Class-1 Rate by {feature}', width=30)
    ax.set_title(wrapped_title, fontsize=20)

    ax.set_xlabel(f'{feature} (0/1)', fontsize=14)
    ax.set_ylabel('Class-1 Probability', fontsize=14)
    ax.set_ylim(0, 1)

# Increase spacing
plt.subplots_adjust(
    wspace=0.35,  # horizontal space between plots
    hspace=0.45  # vertical space between plots
)

plt.tight_layout(rect=[0, 0, 1, 0.97])  # leave extra space at top
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(14, 16))

for i, feature in enumerate(binary_features[4:8]):
    rate = df.groupby(feature)['class'].mean()

    ax = axes[i // 2, i % 2]  # Select subplot

    sns.barplot(x=rate.index.astype(str), y=rate.values, ax=ax)

    # Wrap long title
    wrapped_title = textwrap.fill(f'Class-1 Rate by {feature}', width=30)
    ax.set_title(wrapped_title, fontsize=20)

    ax.set_xlabel(f'{feature} (0/1)', fontsize=14)
    ax.set_ylabel('Class-1 Probability', fontsize=14)
    ax.set_ylim(0, 1)

# Increase spacing
plt.subplots_adjust(
    wspace=0.35,  # horizontal space between plots
    hspace=0.45  # vertical space between plots
)

plt.tight_layout(rect=[0, 0, 1, 0.97])  # leave extra space at top
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 16))

for i, feature in enumerate(binary_features[8:]):
    rate = df.groupby(feature)['class'].mean()

    ax = axes[ i ]  # Select subplot

    sns.barplot(x=rate.index.astype(str), y=rate.values, ax=ax)

    # Wrap long title
    wrapped_title = textwrap.fill(f'Class-1 Rate by {feature}', width=30)
    ax.set_title(wrapped_title, fontsize=20)

    ax.set_xlabel(f'{feature} (0/1)', fontsize=14)
    ax.set_ylabel('Class-1 Probability', fontsize=14)
    ax.set_ylim(0, 1)

# Increase spacing
plt.subplots_adjust(
    wspace=0.35,  # horizontal space between plots
    hspace=0.45  # vertical space between plots
)

plt.tight_layout(rect=[0, 0, 1, 0.97])  # leave extra space at top
plt.show()



plt.figure(figsize=(6, 4))
emp_rate = df.groupby('employment')['class'].mean()

sns.barplot(x=emp_rate.index, y=emp_rate.values)
plt.title('Class-1 Rate by Employment Length')
plt.xlabel('Employment Group')
plt.ylabel('Class-1 Probability')
plt.ylim(0, 1)
plt.tight_layout()
plt.show()


