import numpy as np
 #function to calculate the joint distribution of two words in a text dataset
def joint_distribution_of_word_counts(texts, word0, word1):
    #initializing arrays to store word counts
    count_word0 = np.zeros(len(texts))
    count_word1 = np.zeros(len(texts))
    #counting occurrences of word0 and word1 in each text
    for i, text in enumerate(texts):
        count_word0[i] = text.count(word0)
        count_word1[i] = text.count(word1)
    #etermining the maximum count of each word
    max_count_word0 = int(max(count_word0))
    max_count_word1 = int(max(count_word1))
    #creating an array to store the joint distribution
    Pjoint = np.zeros((max_count_word0 + 1, max_count_word1 + 1))
    #computing the joint distribution
    for i in range(len(texts)):
        #converting the word counts to integers
        x0 = int(count_word0[i])
        x1 = int(count_word1[i])
        #incrementing the corresponding entry in the joint distribution by 1
        Pjoint[x0, x1] += 1
    #normalizing the joint distribution by dividing each entry by the total number of texts
    Pjoint /= len(texts)
    return Pjoint
#function to compute the marginal distribution based on the joint distribution
def marginal_distribution_of_word_counts(Pjoint, index):
    if index == 0:
        return np.sum(Pjoint, axis=1) #computing the sum along axis 1 (for word0)
    elif index == 1:
        return np.sum(Pjoint, axis=0) #computing the sum along axis 0 (for word1)
    else:
        raise ValueError("Invalid index value. Must be 0 or 1.")

#function to calculate the conditional distribution using the joint and marginal distributions
def conditional_distribution_of_word_counts(Pjoint, Pmarginal):
     #creating an array to store the conditional distribution
    Pcond = np.zeros_like(Pjoint)
     #loop over the rows and columns of Pjoint to compute the conditional distribution n handling division by zero errors
    for i in range(Pjoint.shape[0]):
        for j in range(Pjoint.shape[1]):
            #checkiing if the marginal probability is not zero to avoid division by zero
            if Pmarginal[i] != 0:
                #computing the conditional distribution for each entry
                Pcond[i, j] = Pjoint[i, j] / Pmarginal[i]
            else:
                 #in the case if probability is zero, assign nan(not a number) to the corresponding entry
                Pcond[i, j] = np.nan
    return Pcond

#fnction to compute the mean from the distribution
def mean_from_distribution(P):
    return round(np.sum(np.arange(len(P)) * P), 3)  #computing the sum of the products of indices and probabilities
    

#function to compute the covariance from the distribution
def covariance_from_distribution(P):
    m, n = P.shape #obtaining the shape of the distribution matrix
    mu = mean_from_distribution(np.sum(P, axis=1))  #computing the mean by summing the rows of the distribution
    covar = 0 #initializing the covariance variable
    for i in range(m):  #iterating through the rows of the distribution
        for j in range(n):  #iterating through the columns of the distribution
            covar += ((i - mu) * (j - mu) * P[i, j])  #calculating the covariance using the formula

    return round(covar, 3)

#function to calculate the variance from a given distribution
def variance_from_distribution(P):
    #computing the mean using the mean_from_distribution function
    mu = mean_from_distribution(P)
    return round(np.sum(((np.arange(len(P)) - mu) ** 2) * P), 3)

#function to calculate the expected value of a function for a given distribution
def expectation_of_a_function(P, f):
    m, n = P.shape
    #initializing the variable to store the expected value
    expected = 0
    #computing the expected value of the function by iterating over the elements of the distribution
    for i in range(m):
        for j in range(n):
            expected += P[i, j] * f(i, j)  #calculating  the expected  value of the function 
    return round(expected, 3)
