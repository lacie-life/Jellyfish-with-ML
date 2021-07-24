# ------------------------------------------------------------
# Create the scatter plot of Sales Vs cost from the file data
# ------------------------------------------------------------

# Import pyplot
import matplotlib.pyplot as plt

# Open the file in read mode and read lines
f = open('salesdata2.csv','r')
salefile = f.readlines()

# Create the sales List
s_list = []
c_list = []

# Append all the records from the file to the saleslist
for records in salefile:
    sale, cost = records.split(sep=',')
    s_list.append(int(sale))
    c_list.append(int(cost))
    

# Change the chart labels
plt.title("Sales Vs Cost")
plt.xlabel("Sale")
plt.ylabel("Cost")

# Create the scatter plot
plt.scatter(s_list,c_list)
plt.show()





