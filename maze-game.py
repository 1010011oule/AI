# importing the Maze class from the 'maze' module and defaultdict from 'collections'.
from maze import Maze
from collections import defaultdict

# defining a function 'search' that takes a maze and a search method as inputs.
def search(maze, func):
    # creating a dictionary mapping search method names to their respective functions.
    search_functions = {
        "bfs": bfs,
        "ids": ids,
        "astar": astar,
        "astar_four_circles": astar_four_circles,
        "astar_many_circles": astar_many_circles
    }
    # getting the search function based on the provided method name.
    search_func = search_functions.get(func)
    # raising an error if the provided search method is not supported.
    if search_func is None:
        raise ValueError(f"Search method '{func}' is not supported.")
    # calling the selected search function with the maze and return the result.
    return search_func(maze)

# defining the 'bfs' function that takes a maze as input.
def bfs(maze: Maze):
    # getting the start and goal points from the maze
    start, goal = maze.startPoint(), maze.circlePoints()[0]
    # initializing a set to track visited points and a queue for BFS traversal.
    visited, queue = set(), [(start, [start])]
    # continuing until the queue is empty.
    while queue:
        # getting the current point and path from the queue
        current, path = queue.pop(0)
        # if the current point is the goal, return the path
        if current == goal:
            return path
        # if the current point has already been visited, skip it
        if current in visited:
            continue
        # marking the current point as visited
        visited.add(current)
        # adding neighboring points to the queue along with their paths
        queue.extend((n, path + [n]) for n in maze.neighborPoints(*current))
    # returning an empty list if no path is found
    return []

# defining the 'ids' function that takes a maze as input
def ids(maze: Maze):
    # setting the maximum depth for iterative deepening
    MAX_D = 1000
    # getting the start and goal points from the maze
    start, goal = maze.startPoint(), maze.circlePoints()[0]
    
    # defining a recursive function 'dls' for depth-limited search
    def dls(node, depth, visited):
        # if the node has already been visited, return an empty list
        if node in visited:
            return []
        # marking the node as visited
        visited.add(node)
        # if the desired depth is reached and the node is the goal, return the path
        if depth == 0 and node == goal:
            return [node]
        # if the depth limit has not been reached, continue searching.
        if depth > 0:
            return next((path + [node] for n in maze.neighborPoints(*node)
                         for path in [dls(n, depth - 1, visited)] if path), [])
        # returning an empty list if the goal is not found within the depth limit
        return []

    # initializing the depth to 0 and continue until the maximum depth is reached
    depth = 0
    while depth < MAX_D:
        visited = set()  # initializing the set of visited points for each iteration
        result = dls(start, depth, visited)  # performing depth-limited search
        # if a path is found then return it
        if result:
            return result
        depth += 1  # incrementing the depth limit if no path is found
    
    # returning an empty list if no path is found within the maximum depth
    return []

# defining the 'astar' function that takes a maze as input
def astar(maze: Maze):
    # getting the start and goal points from the maze
    start, goal = maze.startPoint(), maze.circlePoints()[0]
    # initializing open_list with the start point and its g and h values
    open_list, closed_list = [(start, 0, 0, [])], set()
    
    # continuing until the open_list is empty
    while open_list:
        # selecting the node with the lowest f value (g + h) from the open_list
        current, g, h, path = min(open_list, key=lambda x: x[1] + x[2])
        # if the current node is the goal, return the path
        if current == goal:
            return path + [current]
        # removing the current node from the open_list and add it to the closed_list
        open_list.remove((current, g, h, path))
        closed_list.add(current)
        # expanding the current node's neighbors and add them to the open_list if not in the closed_list
        open_list.extend((n, g + 1, h + 1, path + [current]) for n in maze.neighborPoints(*current) if n not in closed_list)
    
    # returning an empty list if no path is found
    return []

# defining the 'astar_four_circles' function that takes a maze as input
def astar_four_circles(maze: Maze):
    # getting the start and initial set of goal points (circles) from the maze
    start, goals = maze.startPoint(), maze.circlePoints()
    final_path = []  # Initialize the final path.
    
    # continuing until there are remaining goals
    while goals:
        open_list, closed_list, find_goal, curr_path = [(start, 0, 0, [])], set(), None, []
        # continuing until the open_list is empty
        while open_list:
            # selecting the node with the lowest f value (g + h) from the open_list
            current, g, h, path = min(open_list, key=lambda x: x[1] + x[2])
            # if the current node is one of the remaining goals, update variables
            if current in goals:
                find_goal, curr_path = current, path + [current]
                break
            # removing the current node from the open_list and add it to the closed_list
            open_list.remove((current, g, h, path))
            closed_list.add(current)
            # expanding the current node's neighbors and add them to the open_list if not in the closed_list
            open_list.extend((n, g + 1, 0, path + [current]) for n in maze.neighborPoints(*current) if n not in closed_list)
        
        # ifa goal is found, remove it from the goals and update the start and final path
        if find_goal:
            goals.remove(find_goal)
            start, final_path = find_goal, final_path + curr_path
        else:
            break  # exiting the loop if no goal is found
    
    # returning the final path that visits all targets
    return final_path

# defining the 'mst_w' function that takes a list of points as input
def mst_w(points):
    # defining a function to calculate the Euclidean distance between two points
    def distance(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    # defining functions for finding the root of a set and performing union of two sets
    def find(p, parent):
        while p != parent[p]:
            p = parent[p]
        return p

    def union(p, q, parent, rank):
        rootP = find(p, parent)
        rootQ = find(q, parent)
        if rootP == rootQ:
            return False
        if rank[rootP] < rank[rootQ]:
            parent[rootP] = rootQ
        elif rank[rootP] > rank[rootQ]:
            parent[rootQ] = rootP
        else:
            parent[rootQ] = rootP
            rank[rootP] += 1
        return True

    # creating a list of all possible edges and sort them by distance
    edges = [(i, j, distance(points[i], points[j])) for i in range(len(points)) for j in range(i + 1, len(points))]
    edges.sort(key=lambda x: x[2])
    
    # initializing a parent dictionary and a rank dictionary for each point
    parent, rank = {i: i for i in range(len(points))}, {i: 0 for i in range(len(points))}
    mst_w = sum(edge[2] for edge in edges if union(edge[0], edge[1], parent, rank))
    
    # returning the weight of the minimum spanning tree
    return mst_w

# defining the 'stage3_heuristic' function that takes a current point and a set of goals as input
def stage3_heuristic(current, goals):
    # calculating the nearest distance from the current point to any target
    nearest_distance = min(abs(current[0] - g[0]) + abs(current[1] - g[1]) for g in goals)
    # returning the heuristic value, which is the sum of the nearest distance and the MST weight of remaining targets
    return nearest_distance + mst_w(list(goals))

# defining the 'astar_many_circles' function that takes a maze as input
def astar_many_circles(maze: Maze):
    # getting the start point and create a set of targets from the maze
    start, goals = maze.startPoint(), set(maze.circlePoints())
    all_paths = []  # initializing a list to store all paths
    
    # continuing until there are remaining goals
    while goals:
        open_list, closed_list = [(start, 0, stage3_heuristic(start, goals), [])], set()
        path = []  # Initialize the path for the current iteration.
        
        # continuing until the open_list is empty
        while open_list:
            # selecting the node with the lowest f value (g + h) from the open_list
            current, g, h, path = min(open_list, key=lambda x: x[1] + x[2])
            # if the current node is one of the remaining goals, update variables
            if current in goals:
                all_paths.append(path + [current])
                start = current
                goals.remove(current)
                break
            # removing the current node from the open_list and add it to the closed_list
            open_list.remove((current, g, h, path))
            closed_list.add(current)
            # expanding the current node's neighbors and add them to the open_list if not in the closed_list
            open_list.extend((n, g + 1, 0, path + [current]) for n in maze.neighborPoints(*current) if n not in closed_list)
    
    # returning the concatenated list of all paths, forming a single path that visits all targets
    return [point for path in all_paths for point in path]
