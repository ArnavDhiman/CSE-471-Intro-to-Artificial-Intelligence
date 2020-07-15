# bayesNet.py

import itertools
import util

class CPT():
    """
    A table that represents the conditional probabilities.
    This has two components, the dependencyList and the probTable.
    dependencyList is a list of nodes that the CPT depends on.
    probTable is a table of length 2^n where n is the length of dependencyList.
    It represents the probability values of the truth table created by the dependencyList as the boolean values, in the same order.
    That is to say is the depencyList contains A and B then probTable will have 4 values corresponding to (A, B) = (0, 0), (0, 1), (1, 0), (1, 1) in that order.
    """
    def __init__(self, dependencies, probabilities):
        self.dependencyList = dependencies
        self.probTable = probabilities

class BayesNetwork():
    """
    A network represented as a dictionary of nodes and CPTs
    """
    def __init__(self, network):
        """
        Constructor for the BayesNetwork class. By default it only takes in the network
        Feel free to add things to this if you think they will help.
        """
        self.network = network        
        self.dependencyMap, self.nodes=self.setDependecyMapNodes()
        #print(self.nodes)


    def setDependecyMapNodes(self):
    #    print(network)
        nodes = []
        dependencyMap = {}
        for node in self.network:
            nodes.append(node)
            dependencyList = self.network[node].dependencyList

            if not dependencyList:
                dependencyMap[node] = [dependencyList,{None:self.network[node].probTable[0]}]
            elif len(dependencyList) == 1:
                dependencyMap[node] = [dependencyList, {(0,) : self.network[node].probTable[0], (1,): self.network[node].probTable[1]}]
            elif len(dependencyList) == 2:
                dependencyMap[node] = [dependencyList, {(0,0) : self.network[node].probTable[0], (0,1): self.network[node].probTable[1], (1,0) : self.network[node].probTable[2], (1,1): self.network[node].probTable[3]}]
        nodes.reverse()

        return (dependencyMap, nodes)
    
    def probability(self, key, value, parentProb, probMap):
        prob = probMap[key][0]
        if not len(prob):
            trueProb = probMap[key][1][None]
        else:
            probVal = [parentProb[parent] for parent in prob]
            trueProb = probMap[key][1][tuple(probVal)]
        if value: 
            return trueProb    
        return 1.0-trueProb

    def getBayesNetProb(self, A, Btuple, dependencyMap, nodes):
        totalProb = {}
        for probTF in [0, 1]:
            Btuple[A] = probTF
            totalProb[probTF] = self.findProb(nodes, Btuple, dependencyMap)
        
        total = 0.0
        for prob in totalProb.values():
            total += prob
        for key in totalProb.keys():
            totalProb[key] = totalProb[key]/total

        return totalProb


    def findProb(self, nodes, Btuple, dependencyMap):
        if not len(nodes): 
            return 1

        node = nodes.pop()
        
        if node in Btuple:
            value = self.probability(node, Btuple[node], Btuple, dependencyMap) * self.findProb(nodes, Btuple, dependencyMap)
            nodes.append(node)
            return value
        else:
            total = 0
            Btuple[node] = 1
            total += self.probability(node, Btuple[node], Btuple, dependencyMap) * self.findProb(nodes, Btuple, dependencyMap)
            Btuple[node] = 0
            total +=self. probability(node, Btuple[node], Btuple, dependencyMap) * self.findProb(nodes, Btuple, dependencyMap)
            del Btuple[node]
            nodes.append(node)
            return total
                  


    def singleInference(self, A, B):
        """
        Return the probability of A given B using the Bayes Network. Here B is a tuple of (node, boolean).
        """
        return self.getBayesNetProb(A,{B[0]:B[1]},self.dependencyMap,self.nodes)[1]        
        
    def multipleInference(self, A, observations):
        """
        Return the probability of A given the list of observations.Observations is a list of tuples.
        """
        
        probability = {}
        for i in range(0,len(observations),2):
            probability[observations[i]]=observations[i+1]
        return self.getBayesNetProb(A,probability,self.dependencyMap,self.nodes)[1]