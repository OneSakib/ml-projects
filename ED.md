# Step 1: Importing Required Libraries
We need to install Pandas, NumPy, Matplotlib and Seaborn libraries in python to proceed further.

# Step 2: Reading Dataset
Let's read the dataset using pandas.

# Step 3: Analyzing the Data
1. df.shape(): This function is used to understand the number of rows (observations) and columns (features) in the dataset. This gives an overview of the dataset's size and structure.
2. df.info(): This function helps us to understand the dataset by showing the number of records in each column, type of data, whether any values are missing and how much memory the dataset uses.
3. df.describe().T: This method gives a statistical summary of the DataFrame (Transpose) showing values like count, mean, standard deviation, minimum and quartiles for each numerical column. It helps in summarizing the central tendency and spread of the data.
4. df.columns.tolist(): This converts the column names of the DataFrame into a Python list making it easy to access and manipulate the column names.

# Step 4 : Checking Missing Values
df.isnull().sum(): This checks for missing values in each column and returns the total number of null values per column helping us to identify any gaps in our data.

# Step 5 : Checking for the duplicate values
df.duplicated().sum(): Returns the number of duplicate rows in the dataset.

# Step 6: Univariate Analysis
In Univariate analysis plotting the right charts can help us to better understand the data making the data visualization so important.
1. Bar Plot for evaluating the count of the wine with its quality rate.
2. Kernel density plots help visualize the distribution of data and identify patterns such as skewness and density.
3. Swarm Plot for showing the outlier in the data

# Step 7: Bivariate Analysis
1. Pair Plot for showing the distribution of the individual variables
2. Violin Plot for examining the relationship between alcohol and Quality.
3. Box Plot for examining the relationship between alcohol and Quality

# Step 8: Multivariate Analysis

