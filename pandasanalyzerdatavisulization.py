import pandas as pd
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class SalesDataAnalyzer:
        def __init__(self):
            self.data = None

        def load_data(self):
            print("=====Lord Dataset======")
            path = input("Enter the path of the dataset (CSV File):")
            try:
                self.data = pd.read_csv(path)
                print("Dataset Loaded Successfully!")
                print(self.data)
            except FileNotFoundError:
               print("File not found. Please check the path and try again.")
            except Exception as e:
               print(f"An error occurred: {e}")
            print("="*30)

 
        def first5rows(self):
            try:
               print("\nTop five record:\n")
               fiverecords = self.data.head()
               print(fiverecords)
            
            except Exception as e:
               print(f"An error occurred: {e}")
            print("="*30)
        
        
        def last5rows(self):
            try:
               print("\nLast five record:\n")

               lastfiverecords = self.data.tail()
               print(lastfiverecords)
            
            except Exception as e:
               print(f"An error occurred: {e}")
            print("="*30)
        
        def showcolumnname(self):
            try:
               print("\nColumn name:\n")

               columnname = self.data.columns.tolist()
               print(columnname)
            
            except Exception as e:
               print(f"An error occurred: {e}")
            print("="*30)

        def showdatatype(self):
            try:
               print("\nData type:\n")

               datatype = self.data.dtypes
               print(datatype)
            
            except Exception as e:
               print(f"An error occurred: {e}")
            print("="*30)
            
        def showbasicinfo(self):
            try:
               print("\nBasic Information:\n")

               basicinfo = self.data.info()
               print(basicinfo)
            
            except Exception as e:
               print(f"An error occurred: {e}")
            print("="*30)   

        def explore_data(self):
            print("=====Explore Data======")
            while True:
                print("Select an option:")
                print("1. Display the top 5 rows")
                print("2. Display the last 5 rows")
                print("3. Display column name")
                print("4. Display Datatype")
                print("5. Display the basic info")
                print("6. Back menu")

                e1 = input("Enter your choice(1-6)")

                match e1:
                    case "1":
                        sales.first5rows()
                    case "2":
                        sales.last5rows()
                    case "3":
                        sales.showcolumnname()
                    case "4":
                        sales.showdatatype()
                    case "5":
                        sales.showbasicinfo()
                    case "6":
                        break
        

       

        def filterfile(self):
            try:
                print("\nFilter Record:\n")
                filterinfo = self.data[self.data["sales"] > 100]
                print(filterinfo)
            except Exception as e:
                print(f"An error occurred: {e}")
            print("=" * 30)

        def search_similar(self):
            try:
                column = input("Enter the column to search in (e.g., product, region): ").strip()
                if column not in self.data.columns:
                    print("Invalid column name.")
                    return

                keyword = input("Enter the search keyword: ").strip().lower()
                results = self.data[self.data[column].str.lower().str.contains(keyword)]

                if not results.empty:
                    print(f"\nSearch results for '{keyword}' in '{column}':")
                    print(results)
                else:
                    print("No matching records found.")
            except Exception as e:
                print(f"An error occurred: {e}")
            print("=" * 30)

        def ascendingsort(self):
            try:
                print("\nAscending Order by Sales:\n")
                asorted_data = self.data.sort_values(by='sales', ascending=True)
                print(asorted_data)
            except Exception as e:
                print(f"An error occurred: {e}")
            print("=" * 30)

        def deascendingsort(self):  # Renamed to follow naming convention
            try:
                print("\nDescending Order by Sales:\n")
                dsorted_data = self.data.sort_values(by='sales', ascending=False)
                print(dsorted_data)
            except Exception as e:
                print(f"An error occurred: {e}")
            print("=" * 30)

        def sortfileBy_Sales(self):
            while True:
               print("Select an option")
               print("1. Ascending")
               print("2. Descending")
               print("3. Back to Menu")

               sortchoice = input("Enter your choice: ")

               match sortchoice:
                    case "1":
                      self.ascendingsort()
                    case "2":
                      self.deascendingsort()
                    case "3":
                      break
                   

        def search_sort_filter(self):
            while True:
               print("Select an option:")
               print("1. Filter")
               print("2. Search")
               print("3. Sort")
               print("4. Back to Previous Menu")

               choice = input("Enter your choice: ")

               match choice:
                case "1":
                    self.filterfile()
                case "2":
                    self.search_similar()
                case "3":
                    self.sortfileBy_Sales()
                case "4":
                    break
                   
        def addofsp(self):
            try:
                print("\nSum:\n")
                salessum = self.data['sales'].sum()
                profitsum = self.data['profit'].sum()
                print(f"sales = {salessum}\t profit = {profitsum}")
            except Exception as e:
                print(f"An error occurred: {e}")
            print("=" * 30)


        def meanofsp(self):
            try:
                print("\nMean:\n")
                salesmean = self.data['sales'].mean()
                profitmean = self.data['profit'].mean()
                print(f"sales = {salesmean}\t profit = {profitmean}")
            except Exception as e:
                print(f"An error occurred: {e}")
            print("=" * 30)
        

        def countofsp(self):
            try:
                print("\nCount:\n")
                salescount = self.data['sales'].count()
                profitcount = self.data['profit'].count()
                print(f"sales = {salescount}\t profit = {profitcount}")
            except Exception as e:
                print(f"An error occurred: {e}")
            print("=" * 30)
         
        def aggregate(self):
            while True:
                print("select an option")
                print("1. Sum")
                print("2. Mean")
                print("3. Count")
                print("4. Back to Previous Menu")

                aggeregate_choice = input("Enter your choice:")
                match aggeregate_choice:
                    case "1":
                        sales.addofsp()
                    case "2":
                        sales.meanofsp()
                    case "3":
                        sales.countofsp()
                    case "4":
                        break

        def combine_data(self):
            try:
                print("\nCombine Dataframe:\n")
        
                movie_df = pd.read_csv("movies.csv")

                if self.data is None:
                    print("Sales data is not loaded yet.")
                    return None

                # Define correct column names
                movie_cols = ['title', 'genre', 'year', 'director', 'rating']
                sales_cols = ['date', 'product', 'sales', 'region', 'profit']

                # Check column count before renaming
                if self.data.shape[1] != len(sales_cols):
                    print(f"Sales data has {self.data.shape[1]} columns, expected 5. Cannot rename.")
                    return None

                if movie_df.shape[1] != len(movie_cols):
                    print(f"Movie data has {movie_df.shape[1]} columns, expected 5. Cannot rename.")
                    return None

                # Rename columns
                self.data.columns = sales_cols
                movie_df.columns = movie_cols

                 # Combine both datasets
                combinedata = pd.concat([self.data, movie_df], ignore_index=True)
                print(combinedata)
                print("=" * 30)

                return combinedata

            except Exception as e:
               print(f"An error occurred: {e}")
               return None
            

        def spilt_data(self,combinedata):
            try:
                movie_df = pd.read_csv("movies.csv")
                movie_len = len(movie_df)

              # Split the combined data based on movie row count
                sales_split = combinedata.iloc[:-movie_len].copy().reset_index(drop=True)
                movie_split = combinedata.iloc[-movie_len:].copy().reset_index(drop=True)

              # Sanity check: try to set columns ONLY IF column count is 5
                if sales_split.shape[1] == 5:
                    sales_split.columns = ['date', 'product', 'sales', 'region', 'profit']
                else:
                    print("Warning: Sales split does not have 5 columns. Skipping column renaming for sales data.")

                if movie_split.shape[1] == 5:
                    movie_split.columns = ['title', 'genre', 'year', 'director', 'rating']
                else:
                    print("Warning: Movie split does not have 5 columns. Skipping column renaming for movie data.")

                # Show results
                print("\nSales DataFrame:\n", sales_split)
                print("\nMovie DataFrame:\n", movie_split)

                return sales_split, movie_split

            except Exception as e:
                print(f"An error occurred: {e}")

        def mathematical_operation(self):
            while True:
               print("Select an option:")
               print("1. Filter, Search, Sort")
               print("2. Aggregate Function(sum,mean,count)")
               print("3. Combine Data")
               print("4. Spilt Data")
               print("5. Back to Main Menu")

               maths1choice = input("Enter your choice: ")

               match maths1choice:
                case "1":
                    self.search_sort_filter()
                case "2":
                       sales.aggregate()
                case "3":
                       combinedata = sales.combine_data()
                case "4":
                       sales.spilt_data(combinedata=combinedata)              
                case "5":
                    break
                case _:
                    print("Invalid choice.")
           
        def rowwithmissvalue(self):
            try:
                print("\nRows with missing values:\n")
                missing_rows = self.data[self.data.isnull().any(axis=1)]
                if missing_rows.empty:
                    print("No missing values found.")
                else:
                    print(missing_rows)
                    print(f"\nTotal rows with missing values: {len(missing_rows)}")
            except Exception as e:
                    print(f"An error occurred: {e}")
            print("=" * 40)


        def fillwithmean(self):
            try:
                   print("\nFilling missing values with column mean...\n")
                   before_missing = self.data.isnull().sum()
                   self.data.fillna(self.data.mean(numeric_only=True), inplace=True)
                   after_missing = self.data.isnull().sum()

                   print("Missing values filled with mean successfully.")
                   print("\nMissing values before fill:\n", before_missing)
                   print("\nMissing values after fill:\n", after_missing)
                   print("\nSample data after fill:\n", self.data.head())
            except Exception as e:
                   print(f"An error occurred: {e}")
            print("=" * 40)

        def droprowwithmissvalue(self):
            try:
                print("\nDropping rows with missing values...\n")
                before = len(self.data)
                self.data.dropna(inplace=True)
                after = len(self.data)
                dropped = before - after

                print(f"Dropped {dropped} row(s) containing missing values.")
                print("\nSample data after drop:\n", self.data.head())
            except Exception as e:
                    print(f"An error occurred: {e}")
            print("=" * 40)

        def replacewithmissvalue(self):
                print("\nReplace missing values with a specific value\n")
                value = input("Enter the value to fill missing values: ")

                try:
                     # Try to convert to float for numeric fill
                    value = float(value)
                except ValueError:
                    pass  # Keep as string if not convertible

                try:
                   before_missing = self.data.isnull().sum()
                   self.data.fillna(value, inplace=True)
                   after_missing = self.data.isnull().sum()

                   print("Missing values replaced successfully.")
                   print("\nMissing values before replace:\n", before_missing)
                   print("\nMissing values after replace:\n", after_missing)
                   print("\nSample data after replace:\n", self.data.head())
                except Exception as e:
                   print(f"An error occurred: {e}")
                print("=" * 40)


        def clean_data(self):
            while True:
                print("Select an option")
                print("1. Display rows with missing values")
                print("2. Fill missing values with mean")
                print("3. Drop rows with missing values")
                print("4. Replace missing values with a specific value")
                print("5. Back Menu")
                
                misschoice = input("Enter your choice:")

                match misschoice:
                    case "1":
                        sales.rowwithmissvalue()
                    case "2":
                        sales.fillwithmean()
                    case "3":
                        sales.droprowwithmissvalue()
                    case "4":
                        sales.replacewithmissvalue()
                    case "5":
                        break        


        def statistics_analysis(self):
            try:
               print("\nStatistics Analysis :\n")

               statisaticsinfo = self.data.describe()
               print(statisaticsinfo)
            
            except Exception as e:
               print(f"An error occurred: {e}")
            print("="*30)
        
        def create_pivot_table(self):
            try:
                print("\nCreate Pivot Table:\n")
                print("Available columns:", self.data.columns.tolist())

                 # Get user inputs
                index_col = input("Index column (e.g., 'region','col4'): ").strip()
                columns_col = input("Columns column (e.g., 'product','col2'): ").strip()
                values_col = input("Values column (e.g., 'sales','col3'): ").strip()
                agg_func = input("Aggregation function (sum, mean, count): ").strip().lower()

                  # Validate columns
                cols = self.data.columns
                for col in [index_col,columns_col,values_col]:
                    if col not in cols:
                        print("Invalid columns ")
                        return
                    
                 # Validate aggregation function
                valid_funcs = ['sum', 'mean', 'count']
                if agg_func not in valid_funcs:
                  print(f"Invalid aggregation function.")
                  return

                 # Create pivot table
                pivot = pd.pivot_table(
                     self.data,
                     index=index_col,
                     columns=columns_col,
                     values=values_col,
                     aggfunc=agg_func,
                     fill_value=0
                     )

                print("\nPivot Table:\n", pivot)

            except Exception as e:
                 print(f"An error occurred: {e}")


        def bar_plot(self):
            try:
                print("\nBar Plot:\n")
                print("Available columns:", self.data.columns.tolist())
        
                x_col = input("Enter the column for X-axis (e.g., 'product', 'region'): ").strip()
                y_col = input("Enter the column for Y-axis (e.g., 'sales', 'profit'): ").strip()

                if x_col not in self.data.columns or y_col not in self.data.columns:
                   print("Invalid column names provided.")
                   return
                
                sns.set_style("darkgrid")
                plt.figure(figsize=(10, 6))
                sns.barplot(data=self.data, x=x_col, y=y_col,color='skyblue')
                plt.title(f"Bar Plot of {y_col} by {x_col}")
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.xticks(rotation=45)
                plt.tight_layout()
                self.save_visulization()
                plt.show()
            except Exception as e:
                print(f"An error occurred: {e}")
            
 
        def line_plot(self):
            try:
                print("\nLine Plot:\n")
                print("Available columns:", self.data.columns.tolist())

                x_col = input("Enter the column for X-axis (e.g, 'Date'): ").strip()
                y_col = input("Enter the column for Y-axis (e.g, 'sales','profit','product','region'): ").strip()

                if x_col not in self.data.columns or y_col not in self.data.columns:
                    print("Invalid column names.")
                    return
                
                
                self.data[x_col] = pd.to_datetime(self.data[x_col])
                self.data.sort_values(by=x_col, inplace=True)
                
                 # Basic line plot
                sns.set_style("darkgrid")
                plt.figure(figsize=(10, 5))
                sns.lineplot(data=self.data, x=x_col, y=y_col,color='skyblue')
                plt.title(f'{y_col.capitalize()} vs {x_col.capitalize()}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                self.save_visulization()

                plt.show()

            except Exception as e:
                print(f"An error occurred: {e}")
           

        def scatter_plot(self):
            try:
                print("\nScatter Plot:\n")
                print("Available columns:", self.data.columns.tolist())

                x_col = input("Enter the column for X-axis (e.g, 'sales'): ").strip()
                y_col = input("Enter the column for Y-axis (e.g, 'profit'): ").strip()

                if x_col not in self.data.columns or y_col not in self.data.columns:
                    print("Invalid column names.")
                    return
               # Basic scatter plot

                sns.set_style("darkgrid")
                plt.figure(figsize=(10, 5))
                sns.scatterplot(data=self.data, x=x_col, y=y_col,color='skyblue')
                plt.title(f'{y_col.capitalize()} vs {x_col.capitalize()}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                self.save_visulization()

                plt.show()

            except Exception as e:
                print(f"An error occurred: {e}")
           

        def histo_gram(self):
            try:
                print("\nHistogram:\n")
                print("Available columns:", self.data.columns.tolist())

                x_col = input("Enter the column for X-axis (e.g, 'sales','profit'): ").strip()

                if x_col not in self.data.columns: 
                    print("Invalid column names.")
                    return
               # Basic histogram
                if not pd.api.types.is_numeric_dtype(self.data[x_col]):
                    print(f"The column '{x_col}' is not numeric and cannot be used for a histogram.")
                    return
                
                sns.set_style("darkgrid")
                plt.figure(figsize=(10, 5))
                sns.histplot(data=self.data, x=x_col,bins=24,color='skyblue')
                plt.title(f'{x_col.capitalize()}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                self.save_visulization()

                plt.show()

            except Exception as e:
                print(f"An error occurred: {e}")
           
        def pie_chart(self):
            try:
                print("\nPie Chart:\n")
                print("Available columns:", self.data.columns.tolist())

                label_col = input("Enter the column for labels (e.g., 'region', 'product'): ").strip()
                value_col = input("Enter the numeric column for values (e.g., 'sales', 'profit'): ").strip()

                if label_col not in self.data.columns or value_col not in self.data.columns:
                    print("Invalid column names.")
                    return

                # Aggregate values
                pie_data = self.data.groupby(label_col)[value_col].sum()

                sns.set_palette("pastel")  # Use Seaborn color palette
                plt.figure(figsize=(8, 8))
                plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140)
                plt.title(f'{value_col.capitalize()} Distribution by {label_col.capitalize()}')
                plt.axis('equal')
                plt.tight_layout()
                plt.legend(loc='upper left')
                self.save_visulization()

                plt.show()

            except Exception as e:
                print(f"An error occurred: {e}")
           


        def stack_chart(self):  
            try:
                print("\nStack Plot:\n")
                print("Available columns:", self.data.columns.tolist())

                x_col = input("Enter the column for X-axis (should be categorical): ").strip()
                y_cols_input = input("Enter numeric columns for stacking (comma-separated, e.g., 'sales,profit'): ").strip()
                y_cols = [col.strip() for col in y_cols_input.split(",")]

                if x_col not in self.data.columns or any(col not in self.data.columns for col in y_cols):
                    print("One or more column names are invalid.")
                    return

                stacked_data = self.data.groupby(x_col)[y_cols].sum()

                sns.set_palette("pastel")  # Use Seaborn color palette

                sns.set_style("whitegrid")  # Apply Seaborn style
                plt.figure(figsize=(10, 6))
                plt.stackplot(stacked_data.index, stacked_data.T, labels=stacked_data.columns, alpha=0.8)
                plt.title(f'Stack Plot by {x_col}')
                plt.xlabel(x_col)
                plt.ylabel("Values")
                plt.legend(loc='upper left')
                plt.xticks(rotation=45)
                plt.tight_layout()
                self.save_visulization()

                plt.show()

            except Exception as e:
                print(f"An error occurred: {e}")
           
        
        def visulize_data(self):
            while True:
                print("Select an option:")
                print("=====Data Visulization=====")
                print("1. Bar Plot")
                print("2. Line Plot")
                print("3. Scatter Plot")
                print("4. Pie Chart")
                print("5. Histogram")
                print("6. Stack Plot")
                print("7. Back Menu")

                visulize_choice = input("Enter your choice:")

                match visulize_choice:
                    case "1":
                        sales.bar_plot()
                    case "2":
                        sales.line_plot()
                    case "3":
                        sales.scatter_plot()
                    case "4":
                        sales.pie_chart()
                    case "5":
                        sales.histo_gram()
                    case "6":
                        sales.stack_chart()
                    case "7":
                        break 

        def save_visulization(self):
            save = input("Do you want to save this file (Yes/No): ").strip().lower()
            if save == "yes":
                filename = input("Enter the file name to save the plot (e.g., bar_plot.jpg): ").strip()
            try:
                plt.gcf().savefig(filename)  # get current figure
                print(f"Visualization is saved as {filename} successfully")
            except Exception as e:
                print(f"An error occurred: {e}")
sales = SalesDataAnalyzer()

print("================Data Analysis &  visulization Program==================")
while True:
    print("Select an option:")
    print("1. Load Dataset")
    print("2. Explore Data")
    print("3. Perform Dataframe Operation")
    print("4. Handle Missing Data")
    print("5. Generate Discriptive Statistics")
    print("6. Generate pivot table")
    print("7. Data Visulization")
    
    print("9. Exit")
    print("="*50)


    choice = input("Enter your choice:")

    match choice:
        case "1":
            sales.load_data()
        case "2":
            sales.explore_data()
        case "3":
            sales.mathematical_operation()
        case "4":
            sales.clean_data()
        case "5":
            sales.statistics_analysis()
        case "6":
            sales.create_pivot_table()    
        case "7":
            sales.visulize_data()
       
        case "8":
            print("="*40)
            print("Exit.Thank You for using pandas Analyzer & Visulization")
            print("="*40)
        case _:
            print("Please enter valid choice.")    
