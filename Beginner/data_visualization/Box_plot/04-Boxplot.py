# -------------------------------------------------------------
# Create the box plot of sales data from the file
# -------------------------------------------------------------

# Import pyplot
import matplotlib.pyplot as plt

# Open the file in read mode and read lines
f = open('../salesdata.csv','r')
salefile = f.readlines()

# Create the sales List
sale_list = []

# Append all the records from the file to the saleslist
for records in salefile:
    sale_list.append(int(records))

# Change the chart labels
plt.title("Box Plot of Sales")

# Create the plot
plt.boxplot(sale_list)

# Show the Plot
plt.show()

