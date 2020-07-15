import util
import multiAgents

class CSProblem:
    graph = []
    domains = dict()
    vars = []

    def __init__(self, csp):
        self.vars = csp['variables']
        d = csp['domains']
        for keys in d:
            d[keys] = set(d[keys])
        self.domains = d
        self.graph = csp['constraints']

    def getConstraintGraph(self):
        return self.graph

    def setConstraintGraph(self, arcs):
        self.graph = arcs

    def getDomains(self):
        return self.domains

    def setDomains(self, dom):
        self.domains = dom

    def getVars(self):
        return self.vars

    def setVars(self, v):
        self.vars = v

    def runAC3(self):
        return multiAgents.arcConsistencyCSP(self)
