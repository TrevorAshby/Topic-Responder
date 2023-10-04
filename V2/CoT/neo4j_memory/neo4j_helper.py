class client_Node():

    def __init__(self, node_type, props):
        self.node_type = node_type
        self.props = props

    def getType(self):
        return self.node_type

    def getProps(self):
        return self.props

    def __str__(self):
        s = "Types: " + str(self.node_type) + "\n" + "Props: " + str(self.props)
        return s
    

# Neo4J for knowledge graphs
#from client_Node import client_Node
class client_Edge():

    def __init__(self, n1, relType, n2):
        self.n1 = n1
        self.relType = relType
        self.n2 = n2

    def getNode1(self):
        return self.n1

    def getNode2(self):
        return self.n2

    def getRel(self):
        return self.relType

    def __str__(self):
        s = "Node 1\n" + str(self.n1) + "\n" + self.relType + "\n" + "Node 2\n" + str(self.n2)
        return s
    

class DAOInterface:
    def __init__(self):
        pass

    def close(self):
        pass

    def query(self):
        pass

    def createNode(self):
        pass

    def createEdge(self):
        pass

    def getNode(self):
        pass

    def getEdge(self):
        pass

    def updateNode(self):
        pass

    def updateEdge(self):
        pass

    def deleteNode(self):
        pass

    def deleteEdge(self):
        pass

from neo4j import GraphDatabase
import re

# For info on Cypher Query Language, see
# https://neo4j.com/developer/cypher/intro-cypher/

# Make sure to pip install neo4j


class Neo4jDAO(DAOInterface):
    def __init__(self, uri, user, pwd):
        super().__init__()
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            print("Failed to create the driver:", e)

    def close(self):
        if self.__driver is not None:
            self.__driver.close()

    def query(self, query, db=None):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try:
            session = self.__driver.session(database=db) if db is not None else self.__driver.session()
            response = list(session.run(query))
        except Exception as e:
            print("Query failed:", e)
        finally:
            if session is not None:
                session.close()
        return response
    
    def createNode2(self, node_type, node_dict):
        dict_str = '{'
        for i, key in enumerate(node_dict.keys()):
            print(key)
            if i != 0 and i != len(node_dict.keys()):
                dict_str += ', '
            dict_str += key + ': \'' + node_dict[key] + '\''
            
        dict_str += '}'
        the_str = f"CREATE (n:{node_type} " + str(dict_str) + ")"
        # 'CREATE (n:Person {name: \'Andy\', title: \'Developer\'})'
        return self.query(the_str)
    
    def createEdge2(self, src_name, src_type, trg_name, trg_type, rel_type, two_way=False):
        the_str = f"MATCH (a:{src_type}), (b:{trg_type}) WHERE a.name = \'{src_name}\' AND b.name = \'{trg_name}\'" 
        the_str += f"CREATE (a)-[:{rel_type}]->(b)"
        if two_way:
            the_str += f"CREATE (b)-[:{rel_type}]->(a)"
        the_str += "RETURN *"
        return self.query(the_str)

