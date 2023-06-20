import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Specify the file path
file_path = 'adult.data'  # Replace with the actual file path

# Read the file
with open(file_path, 'r') as file:
    data = file.readlines()

# Define the column names
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
                'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
                'native_country', 'income']  

# Extract data rows
rows = [row.strip().split(',') for row in data]

# Create the DataFrame
df = pd.DataFrame(rows, columns=column_names)

# Access and manipulate the data
print(df.head())  # Print the first few rows of the DataFrame

print(df.shape)

print(df.info()) #to check details about the dataset

print(df.duplicated().sum()) #to check for duplicates 

df = df.drop_duplicates()#remove the 24 duplicates

print(df.duplicated().sum())

#Crosschecking for null values
print(df.isnull().sum())

df.dropna(inplace=True)#drop the null values

print(df.isnull().sum())

print(df.dtypes)


columns_to_convert = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

# Loop through each column and convert the data type to int
for column in columns_to_convert:
    df[column] = df[column].astype(int)

# Verify the updated data types
print(df.dtypes)

plt.figure(figsize=(35, 6))

plt.subplot(1, 3, 1)
plt1 = sns.countplot(data=df, x='education', hue='income')
plt.title('Distribution Of Income with education')
plt.xlabel('Education')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(15, 10))
sns.histplot(data=df, x='education', hue='sex', multiple='stack', palette='Set2')

# Set the x-axis label
plt.xlabel('Education and Gender')
plt.xticks(rotation=60)
plt.show()

plt.figure(figsize=(35, 6))

plt.subplot(1, 3, 1)
plt1 = sns.countplot(data=df, x='occupation', hue='income', palette='Set1')
plt.title('Distribution Of Income In Terms of Occupation')
plt.xlabel('Occupation')
plt.ylabel('Frequency')
plt.xticks(rotation=60)
plt.show()

plt.figure(figsize=(30, 10))
sns.histplot(data=df, x='occupation', hue='sex', multiple='stack', palette='Set3')
plt.xticks(rotation=60)
# Set the x-axis label
plt.xlabel('Occupation And Gender')

# Show the plot
plt.show()

# Count the frequency of each gender
gender_counts = df['sex'].value_counts()


# Show the plot
plt.show()


marital_counts = df['marital_status'].value_counts()
fig, ax = plt.subplots(figsize=(15, 10))
# Create a pie chart
plt.pie(marital_counts, labels=marital_counts.index, autopct='%1.1f%%')

# Set the title
plt.title('Marital Status Distribution')

# Show the plot
plt.show()


plt.figure(figsize=(17, 6))
# Create a box plot
sns.boxplot(x='occupation', y='hours_per_week', data=df)

# Set the title and labels
plt.title('Work Hours Distribution by Occupation')
plt.xlabel('Occupation')
plt.ylabel('Hours per Week')

# Rotate x-axis labels for better readability
plt.xticks(rotation=60)

# Show the plot
plt.show()


# Create the line graph
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='occupation', y='hours_per_week', hue='race', linewidth=2, antialiased=False)

# Set labels and title
plt.xlabel('Occupation')
plt.ylabel('Work Hours per Week')
plt.title('Work Hours by Occupation with Race')

# Rotate x-axis labels for better visibility
plt.xticks(rotation=45)

# Show the plot
plt.show()


plt.figure(figsize=(20, 6))
# Create a histogram
plt.hist(df['hours_per_week'], bins=10)

# Set the title and labels
plt.title('Work Hours Distribution')
plt.xlabel('Hours per Week')
plt.ylabel('Frequency')

# Show the plot
plt.show()

grouped_data = df.groupby(['marital_status', 'income']).size().unstack()
plt.figure(figsize=(20, 6))
# Create a bar plot
grouped_data.plot(kind='bar', stacked=True)

# Set the title and labels
plt.title('Income Distribution by Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Count')

# Show the plot
plt.show()


grouped_data = df.groupby(['relationship', 'income']).size().unstack()

# Plot pie charts for each relationship category
for relationship in grouped_data.index:
    plt.figure()
    plt.pie(grouped_data.loc[relationship], labels=grouped_data.columns, autopct='%1.1f%%')
    plt.title(f'Income Distribution for {relationship}')
    plt.axis('equal')

# Show the plots
plt.show()




