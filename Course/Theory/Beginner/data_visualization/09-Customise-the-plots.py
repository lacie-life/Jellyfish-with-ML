# ------------------------------------------------------------
# Customise various elements of the plot for visual appeal.
# ------------------------------------------------------------

# import pyplot
import matplotlib.pyplot as plt

# Open the file in read mode and read lines
f = open('salesdata2.csv','r')
salefile = f.readlines()

# Create the sales List
sale_list = []
s_list = []
c_list = []

# Append all the records from the file to the saleslist
for records in salefile:
    sale, cost = records.split(sep=',')
    s_list.append(int(sale))
    c_list.append(int(cost))
    
# Create list of lists
sale_list.append(s_list)
sale_list.append(c_list)

# Scatter Plot
plt.subplot(2,3,1)
plt.title("Sales Vs Cost")
plt.xlabel("Sale")
plt.ylabel("Cost")


# Markers - o, s, ^, *, +, X, D
# https://matplotlib.org/api/markers_api.html#module-matplotlib.markers

# c - color options - b,g,r,c,m,y,k,w
# c - color options - float in quotes for greyscale '0.80'
# c - color options - HTML hex code such as '#FF5733'
plt.scatter(s_list,c_list,         \
            marker='*',            \
            s=100, c='#FF5733')

# Boxplot
plt.subplot(2,3,2)
plt.title("Box Plot of Sales")
plt.ylabel("USD")
plt.boxplot(sale_list,                                               \
            patch_artist=True,                                       \
            boxprops=dict(facecolor='g', color='r', linewidth=2),    \
            whiskerprops=dict(color='r', linewidth=2),               \
            medianprops=dict(color='w', linewidth=1),                \
            capprops=dict(color='k', linewidth=2),                   \
            flierprops=dict(markerfacecolor='r', marker='o', markersize=7))


# Histogram of sales
plt.subplot(2,3,3)
plt.title("Histogram of Sales")
plt.ylabel("USD")
plt.hist(s_list, bins=5, rwidth=0.9, color='c')

# lineplot of stock
x_days  = [1,2,3,4,5]
y_price1 = [9,9.5,10.1,10,12]

plt.subplot(2,3,4)
plt.title("Stockprice History")
plt.ylabel("Price")
plt.xlabel("Day")
plt.plot(x_days, y_price1, color='green', marker='o', markersize=10, linewidth=3, linestyle='--')

# Bar Chart of temperature variation
x_cities = ['NewYork', 'London', 'Dubai', 'New Delhi','Tokyo']
y_temp = [75,65,105,98,90]

plt.subplot(2,3,5)
plt.title("Temperaure Variation")
plt.xlabel("Cities")
plt.ylabel("Temperature")
plt.xticks(rotation='45')
plt.bar(x_cities,y_temp, color=['r','g', 'c', 'y', 'm'])

# tight_layout to avoid overlap
plt.tight_layout()

# Save the figure as png picture
plt.savefig('images/01custom.png')

# Show plots
plt.show()







