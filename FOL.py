import copy, queue 

# this function is for standardizing variables in the rules
def standardize_variables(nonstandard_rules):
    standardized_rules = {}  # dictionary to store the standardized rules
    variables = []  # list to store the generated variable nams
    # iterating through each rule and standardizing variables
    for i, key in enumerate(nonstandard_rules):
        rule = nonstandard_rules[key]  #extracting the current rule
        standardized_rule = copy.deepcopy(rule)  # creating a deep copy of the rule
        #replacing smthing with unique variables
        for j, antecedent in enumerate(standardized_rule['antecedents']):
            for k, term in enumerate(antecedent):
                if term == 'something':
                    new_variable = f'x{i:04d}'  # generating unique variable name
                    variables.append(new_variable)  # adding the variable name to the list
                    standardized_rule['antecedents'][j][k] = new_variable  # updating the rule
        for j, term in enumerate(standardized_rule['consequent']):
            if term == 'something':
                new_variable = f'x{i:04d}'  # generating unique variable name
                variables.append(new_variable)  # adding the variable name to the list
                standardized_rule['consequent'][j] = new_variable  # updating the rule
        standardized_rules[key] = standardized_rule  # storing the standardized rule
    return standardized_rules, variables  # returning the standardized rules and the list of variables

# this function to unify the query and datum
def unify(query, datum, variables):
    unification = []  # list to store the unified query
    subs = {}  # dictionary to store the substitutions
    # checking if query and datum are of equal length and the last elements are the same
    if len(query) != len(datum) or query[-1] != datum[-1]:
        return None, None  # returning none if unification fails
    # performing unification for each term in query and datum
    for q, d in zip(query, datum):
        if q in variables and d in variables:
            # if both terms are variables and not already unified, perform substitution
            if q in subs and subs[q] != d:
                return None, None  # returning none if unification fails
            subs[q] = d  # storing the substitution
            unification.append(d)  # adding the term to the unified query
        elif q != d:
            return None, None  # returning none if unification fails
        else:
            unification.append(q)  # adding the term to the unified query
    return unification, subs  # returning the unified query and the substitution dictionary

# function to apply a rule to a set of goals
def apply(rule, goals, variables):
    applications = []  # list to store the modified rules after application
    goalsets = []  # list to store the modified goals after application
    # iterate through each goal and attempt to apply the rule
    for goal in goals:
        unification, subs = unify(rule['consequent'], goal, variables)  # Attempt unification
        # if unification is possible, modify the rule and goals accordingly
        if unification is not None:
            new_rule = copy.deepcopy(rule)  # creating a deep copy of the rule
            new_goals = copy.deepcopy(goals)  # creating a deep copy of the goals
            # modifying the antecedents of the rule using the obtained substitutions
            for i, antecedent in enumerate(new_rule['antecedents']):
                for j, term in enumerate(antecedent):
                    if term in subs:
                        new_rule['antecedents'][i][j] = subs[term]
            new_rule['consequent'] = unification  # updating the rule consequent
            new_goals.remove(goal)  # removing the unified goal from the goals
            # modifying the antecedents of the rule using the obtained substitutions
            for i, antecedent in enumerate(new_rule['antecedents']):
                for j, term in enumerate(antecedent):
                    if term in subs:
                        new_rule['antecedents'][i][j] = subs[term]
            applications.append(new_rule)  # adding the modified rule to the applications list
            goalsets.append(new_goals)  # adding the modified goals to the goalsets list
    return applications, goalsets  # returning the modified rules and goals

# function to perform backward chaining
def backward_chain(query, rules, variables):
    queue = []  # initializing a queue to store the queries
    queue.append((query, []))  # adding the initial query and an empty proof to the queue
    while queue:  #looping through the queue until it is empty
        q, proof = queue.pop(0)  #extracting the query and proof from the front of the queue
        # checking for cyclic proof
        if q in proof:
            continue  #continuing to the next iteration if cyclic proof is found
        proof.append(q)  #adding the query to the proof
        # if the query is true, return the proof
        if all([q[-1] for q in proof]):
            return proof  # returning the proof if the query is true
        # trying to apply each rule to the current query and add new proofs to the queue
        for rule_id, rule in rules.items():
            applications, goalsets = apply(rule, [q], variables)  # applying the rule to the query
            for i in range(len(applications)):
                new_proof = proof.copy()  # creating a copy of the current proof
                new_proof.extend(applications[i]['antecedents'])  # adding the rule antecedents to the proof
                queue.append((applications[i]['consequent'], new_proof))  # adding the new query and proof to the queue
    return None  # returning none if no proof is found
