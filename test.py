import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   
from sklearn.linear_model import LinearRegression


vs=pd.read_csv('vgsales.csv')




GamesCount=pd.DataFrame(vs['Year'].value_counts(ascending=True))
GamesCount.reset_index(inplace=True)
GamesCount.columns=['Year', 'count']

GamesCount=GamesCount.sort_values(by=['Year'],ascending=True)

top5Sales=vs.groupby('Name').sum().sort_values(by='Global_Sales', ascending=False)
top5salesname=top5Sales.head(5).index
top5Salesmoney=top5Sales.head(5)['Global_Sales'].values

# Setting up the figure and subplots

plt.figure(figsize=(18, 15))
plt.suptitle('Video Game Sales Analysis', fontsize=16)
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.style.use('ggplot')
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 8)

# Plotting the distribution of video games by platform

plt.subplot(3,3,1)
sns.histplot(vs, x='Platform', hue='Country',multiple='stack', alpha=0.7)
plt.title('Number of Video Games by Platform')
plt.xlabel('Platform')
plt.ylabel('Number of Games')
plt.xticks(rotation=60)
plt.tight_layout()

# Plotting the distribution of video game sales by genre

plt.subplot(3,3,2)
plt.pie(vs['Genre'].value_counts(), labels=vs['Genre'].unique(), autopct='%1.1f%%', startangle=70)
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.
plt.tight_layout()

# Plotting the number of video games released by year

plt.subplot(3,3,3)
plt.plot(GamesCount['Year'],GamesCount['count'])
plt.title('Number of Video Games Released by Year')
plt.xlabel('Year')
plt.ylabel('Number of Games Released')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()

# Plotting the top 5 video games by global sales

plt.subplot(3,3,4)
sns.barplot(x=top5salesname, y=top5Salesmoney, palette='viridis')
plt.title('Top 5 Video Games by Global Sales')
plt.xlabel('Game Name')
plt.ylabel('Global Sales (in millions)')
plt.xticks(rotation=45)
plt.tight_layout()

plt.subplot(3,3,5)
sns.barplot(x='Country', y='Global_Sales', data=vs, palette='Set2')
plt.title('Global Sales by Country')
plt.xlabel('Country')
plt.ylabel('Global Sales (in millions)')
plt.xticks(rotation=45)
plt.tight_layout()
# Plotting the distribution of video game sales by region

ax = plt.subplot(3, 3, 6)
sns.barplot(data=vs, x='Country', y='Global_Sales', hue='Genre', alpha=0.7, ax=ax)
ax.set_title('Video Game Sales by Region and Genre')
ax.set_xlabel('Country')
ax.set_ylabel('Global Sales (in millions)')
ax.tick_params(axis='x', rotation=46)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=3, title='Genre')

# Plotting the distribution of video game sales by year and genre


sales_by_year = vs[['Year', 'Global_Sales']].dropna()

sales_by_year = sales_by_year.groupby('Year').sum().reset_index()

sales_by_year = sales_by_year[sales_by_year['Year'].apply(lambda x: pd.notnull(x) and float(x).is_integer())]

sales_by_year['Year'] = sales_by_year['Year'].astype(int)

if sales_by_year.empty:
    print("No valid year and sales data available for prediction.")
else:
    # Fit linear regression
    X = sales_by_year[['Year']]
    y = sales_by_year['Global_Sales']
    model = LinearRegression()
    model.fit(X, y)

    # Predict for existing and future years
    future_years = np.arange(sales_by_year['Year'].min(), sales_by_year['Year'].max() + 6).reshape(-1, 1)
    predicted_sales = model.predict(future_years)

    # Plot actual and predicted sales
    plt.subplot(3,3,7)
    plt.plot(sales_by_year['Year'], sales_by_year['Global_Sales'], label='Actual Sales', marker='o')
    plt.plot(future_years.flatten(), predicted_sales, label='Predicted Sales', linestyle='--', color='red')
    plt.title('Actual and Predicted Global Sales by Year')
    plt.xlabel('Year')
    plt.ylabel('Global Sales (in millions)')
    plt.legend()
    plt.tight_layout()

# Saving the figure

plt.savefig('video_game_sales_analysis.png', dpi=300)
plt.show()
