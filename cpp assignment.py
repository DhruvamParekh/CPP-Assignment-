#!/usr/bin/env python
# coding: utf-8

# In[9]:


def F(n):
    for i in range(1, n + 1):
        sequence = ""
        for j in range(i):
            sequence += chr(ord('A') + j)
        full_row = sequence + sequence[-2::-1]
        dashes = '-' * (n - i)
        print(dashes + full_row + dashes)



F(10)
F(6)


# In[1]:


def F(d):
    for key in sorted(d.keys()):
        x, y = d[key]
        print(f"-{key}-, -{x}-, -{y}-")
    
    for key in sorted(d, key=lambda k: d[k][0], reverse=True):
        x, y = d[key]
        print(f"-{key}-, -{x}-, -{y}-")
    
    for key in sorted(d, key=lambda k: d[key][1]):
        x, y = d[key]
        print(f"-{key}-, -{x}-, -{y}-")
    pass
print("Fisrt result",F({1 : (1, 2), 2 : (-1, 4), 5 : (-4, 3), 4 : (2, 3)}))
print("Second result",F({-8 : (4, 2), 6 : (-3, 4), 7 : (2, 1), 5 : (9, -10)}))


# In[11]:


import pandas as pd

df = pd.read_csv("student_records.csv")
print(df.head(10))


# In[17]:


letter_to_points = {'AP': 10, 'AA': 10, 'AB': 9, 'BB': 8, 'BC': 7, 'CC': 6}
df['GradePoints'] = df['grade'].map(letter_to_points)
df['WeightedCredits'] = df['GradePoints'] * df['credit']
student_group = df.groupby('roll_number')
total_credits = student_group['credit'].sum()
cpi = student_group['WeightedCredits'].sum() / total_credits
result = pd.DataFrame({'TotalCredits': total_credits, 'CPI': cpi})
print(result)


# In[21]:


core_credits = df[df['course_type'] == 'core'].groupby('roll_number')['credit'].sum()
dept_credits = df[df['course_type'] == 'department_elective'].groupby('roll_number')['credit'].sum()
flexible_credits = df[df['course_type'] == 'flexible_elective'].groupby('roll_number')['credit'].sum()
hasmed_credits = df[df['course_type'] == 'hasmed_elective'].groupby('roll_number')['credit'].sum()

eligible_students = core_credits[core_credits >= 20].index.intersection(
    dept_credits[dept_credits >= 15].index).intersection(
    flexible_credits[flexible_credits >= 10].index).intersection(
    hasmed_credits[hasmed_credits >= 5].index)

result = df[df['roll_number'].isin(eligible_students)][['roll_number']].drop_duplicates()
print(result if not result.empty else "No students meet the graduation requirements.")


# In[23]:


minor_credits = df[df['course_type'] == 'minor'].groupby('roll_number')['credit'].sum()
eligible_students_minor = minor_credits[minor_credits >= 10].index
result = df[df['roll_number'].isin(eligible_students_minor)][['roll_number']].drop_duplicates()
print(result if not result.empty else "No students completed a minor.")


# In[25]:


honours_credits = df[df['course_type'] == 'honours'].groupby('roll_number')['credit'].sum()
core_credits = df[df['course_type'] == 'core'].groupby('roll_number')['credit'].sum()
eligible_students_honours = honours_credits[honours_credits >= 10].index.intersection(
    core_credits[core_credits >= 20].index)
result = df[df['roll_number'].isin(eligible_students_honours)][['roll_number']].drop_duplicates()
print(result if not result.empty else "No students completed honours.")


# In[27]:


import numpy as np
from scipy.optimize import minimize
def objective_function(vars):
    x, y = vars
    return 2 * (x - y - 3)**2 + 4 * (x + 2*y + 1)**4
constraints = [
    {'type': 'ineq', 'fun': lambda vars: vars[0] - vars[1] + 3},  # x - y >= -3
    {'type': 'ineq', 'fun': lambda vars: 5 - ((vars[0] + 2)**2 + (vars[1] + 1)**2)}  # (x+2)^2 + (y+1)^2 <= 5
]
initial_guess = [0, 0]
result = minimize(objective_function, initial_guess, constraints=constraints)
print("Optimal solution (x, y):", result.x)
print("Minimum value of the function:", result.fun)


# In[29]:


import numpy as np
from scipy.integrate import quad
def f(t):
    x = np.sqrt(3) * np.cos(t)
    y = np.sqrt(3) * np.sin(t)
    dx_dt = -np.sqrt(3) * np.sin(t)
    dy_dt = np.sqrt(3) * np.cos(t)
    return (x**2 + y**4) * np.sqrt(dx_dt**2 + dy_dt**2)
result, _ = quad(f, 0, 2 * np.pi)
print(result)


# In[3]:


import time  # To time the execution
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


### TODO 1
### Load data from data_path
### Return: np array of size Nx2
def load_data(data_path):
    # Assuming the file contains rows of "x, y" coordinates separated by commas
    data = np.loadtxt(data_path, delimiter=",")
    return data


# In[7]:


### TODO 2.1
### Initialize the centers by selecting K random points or using provided centers
### Return: np array of size Kx2
def initialise_centers(data, K, init_centers=None):
    if init_centers is None:
        indices = np.random.choice(data.shape[0], K, replace=False)
        centers = data[indices]
    else:
        centers = np.array(init_centers)
    return centers


# In[9]:


### TODO 2.2
### Initialize the labels to ones of size N
### Return: np array of size N
def initialise_labels(data):
    labels = np.ones(data.shape[0], dtype=int)
    return labels


# In[11]:


### TODO 3.1: E step
### Calculate distances of each point to each center
### Return: np array of size NxK
def calculate_distances(data, centers):
    # Use broadcasting to calculate pairwise distances
    distances = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
    return distances


# In[13]:


### TODO 3.2: E step
### Assign the label of the nearest center for each data point
### Return: np array of size N
def update_labels(distances):
    # Find the index of the minimum distance for each point
    labels = np.argmin(distances, axis=1)
    return labels


# In[15]:


### TODO 4: M step
### Update the centers to the mean of the points assigned to each cluster
### Return: np array of size Kx2
def update_centers(data, labels, K):
    centers = np.array([data[labels == k].mean(axis=0) for k in range(K)])
    return centers


# In[17]:


### TODO 6: Check convergence
### Check if labels have changed from the previous iteration
### Return: True / False
def check_termination(labels1, labels2):
    return np.array_equal(labels1, labels2)


# In[19]:


### DON'T CHANGE THIS FUNCTION
def kmeans(data_path: str, K: int, init_centers):
    '''
    Input:
        data (type str): path to the file containing the data
        K (type int): number of clusters
        init_centers (type numpy.ndarray): initial centers. shape = (K, 2) or None
    Output:
        centers (type numpy.ndarray): final centers. shape = (K, 2)
        labels (type numpy.ndarray): label of each data point. shape = (N,)
        time (type float): time taken by the algorithm to converge in seconds
    N is the number of data points each of shape (2,)
    '''
    data = load_data(data_path)    
    centers = initialise_centers(data, K, init_centers)
    labels = initialise_labels(data)

    start_time = time.time()  # Time stamp 

    while True:
        distances = calculate_distances(data, centers)
        labels_new = update_labels(distances)
        centers = update_centers(data, labels_new, K)
        if check_termination(labels, labels_new): break
        else: labels = labels_new
 
    end_time = time.time()  # Time stamp after the algorithm ends
    return centers, labels, end_time - start_time


# In[21]:


### TODO 7: Visualization
def visualise(data_path, labels, centers):
    data = load_data(data_path)

    # Scatter plot of the data points
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

    # Set titles and labels
    plt.title('K-means clustering')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Save the plot as 'kmeans.png'
    plt.savefig('kmeans.png')

    ## DO NOT CHANGE THE FOLLOWING LINE
    return plt


# In[27]:


# Run the code to generate the plot
if __name__ == "__main__":
    data_path = 'spice_locations.txt'  # Change to actual path of the data file
    K, init_centers = 2, None
    centers, labels, time_taken = kmeans(data_path, K, init_centers)
    print('Time taken for the algorithm to converge:', time_taken)
    visualise(data_path, labels, centers)


# In[ ]:




