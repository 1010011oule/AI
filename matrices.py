# importing necessary libraries
import numpy as np
import utils  # assuming utils module is imported and defined elsewhere

# setting a small value for epsilon
epsilon = 1e-3
# initializing variables M and N
M, N = 0, 0

# function to calculate the next position based on the current position and action
def move(r, c, a, model):
    next_r, next_c = r, c
    
    # updating next position based on the action
    if a == 0 and c > 0:  # Left
        next_c = c - 1
    elif a == 1 and r > 0:  # Up
        next_r = r - 1
    elif a == 2 and c < N - 1:  # Right
        next_c = c + 1
    elif a == 3 and r < M - 1:  # Down
        next_r = r + 1
        
    # checking if the next position is not a wall
    if not model.W[next_r, next_c]:
        return next_r, next_c
    return r, c

# function to compute the transition matrix
def compute_transition_matrix(model):
    """
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    """
    global M, N
    M, N = model.M, model.N
    P = np.zeros((M, N, 4, M, N))

    # loop through each cell and action to calculate transition probabilities
    for r in range(M):
        for c in range(N):
            if model.T[r, c]:
                continue  # terminal states have no transitions

            for a in range(4):
                next_r, next_c = move(r, c, a, model)
                x, y = move(r, c, (a - 1) % 4, model)
                if not model.W[x, y]:
                    P[r, c, a, x, y] += model.D[r, c, 1]
                x, y = move(r, c, (a + 1) % 4, model)
                if not model.W[x, y]:
                    P[r, c, a, x, y] += model.D[r, c, 2]

                # checkin if the next state is not a wall
                if not model.W[next_r, next_c]:
                    P[r, c, a, next_r, next_c] += model.D[r, c, 0]  # seting transition probability
                else:
                    continue  # stay in the current state if hitting a wall

    # handle terminal states
    P[model.T, :, :, :] = 0
    print(P)
    return P

# function to update the utility function
def update_utility(model, P, U_current):
    """
    Parameters:
    model - The MDP model returned by load_MDP()
    P - The precomputed transition matrix returned by compute_transition_matrix()
    U_current - The current utility function, which is an M x N array

    Output:
    U_next - The updated utility function, which is an M x N array
    """

    M, N = model.M, model.N
    U_next = np.zeros((M, N))

    # loop through each cell to update utility
    for r in range(M):
        for c in range(N):
            max_expected_utility = 0
            # loop through each action to calculate expected utility
            for a in range(4):
                expected_utility = np.sum(P[r, c, a, :, :] * U_current)
                max_expected_utility = max(max_expected_utility, expected_utility)

            # updating the utility function for the current cell
            U_next[r, c] = model.R[r, c] + model.gamma * max_expected_utility

    return U_next

# function for value iteration
def value_iteration(model, P):
    """
    Parameters:
    model - The MDP model returned by load_MDP()
    P - The precomputed transition matrix returned by compute_transition_matrix()

    Output:
    U - The utility function, which is an M x N array
    """

    M, N = model.M, model.N
    U_current = np.zeros((M, N))

    # iteratively update the utility function until convergence
    while True:
        U_next = update_utility(model, P, U_current)

        # checking for convergence
        if np.max(np.abs(U_next - U_current)) < epsilon:
            break

        U_current = U_next

    return U_next
