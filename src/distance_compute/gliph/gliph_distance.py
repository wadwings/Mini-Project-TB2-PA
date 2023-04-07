

def levenshtein_distance(s1, s2):
    # Initialize matrix with zeros
    rows = len(s1)+1
    cols = len(s2)+1
    distance = [[0 for col in range(cols)] for row in range(rows)]

    # Populate matrix with initial values
    for row in range(1, rows):
        distance[row][0] = row
    for col in range(1, cols):
        distance[0][col] = col

    # Calculate edit distance
    for col in range(1, cols):
        for row in range(1, rows):
            if s1[row-1] == s2[col-1]:
                cost = 0
            else:
                cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,   # deletion
                                     distance[row][col-1] + 1,   # insertion
                                     distance[row-1][col-1] + cost) # substitution

    return distance[rows-1][cols-1]

if __name__ == '__main__':
    print(levenshtein_distance(s1='CIVRAPGRADMRF', s2='CASSYLPGQGDHYSNQPQHF'))