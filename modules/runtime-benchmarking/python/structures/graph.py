#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 2021/09/13

'''

from models.trap_model import Trap

class DynamicArrayGraph(object):
    '''
    Simple graph class for defining traps connected by the dynamic array.

    Attributes:
    ----
        array_object_by_idx : dict
            dict indexing trap objects corresponding to graph vertices.
        
        edges_by_idx : dict
            dict indexing the edges of a vertex.

    '''


    def __init__(self, array_object_by_idx: dict={}, edges_by_idx: dict={}):
        ''' 
        Initialize graph object.

        Args:
        ----
            edges_by_idx (dict) : dictionary indexing edges of each vertex
            array_object_by_idx (dict) : dictionary indexing each trap in dynamic array

        '''

        self.array_object_by_idx = array_object_by_idx
        self.edges_by_idx = edges_by_idx

    def get_edges(self, vertex):
        ''' 
        Returns set of all vertices with an edge to the vertex provided.

        Returns:
        ----
            (list of sets) : list of two-element sets of vertices with an edge between them

        '''

        return self.edges_by_idx[vertex]

    def set_edges(self, edges: dict):
        ''' 
        Set edges of the graph. In format of vertex_label : [edge1, edge2, ...].

        '''

        self.edges_by_idx = edges

    def get_all_vertices(self):
        ''' 
        Returns all vertices within graph as a set.

        Returns: 
        ----
            (set) : set of vertices in graph

        '''

        return set(self.edges_by_idx.keys())

    def get_all_edges(self):
        '''
        Return all edges within a graph as a set.
        
        Returns:
            (set) : set of edges within graph

        '''

        return self._generate_edges()

    def add_vertex(self, vertex: int, trap: Trap):
        ''' 
        Add vertex to graph dictionaries.

        Args:
            vertex (int) : label of vertex
            trap (Trap) : trap object defining coordinates in space and occupation state

        '''

        if vertex not in self.array_object_by_idx:
            self.array_object_by_idx[vertex] = trap
            self.edges_by_idx[vertex] = {}

    def add_edge(self, edge: tuple):
        ''' 
        Add edge fo graph dictionary.

        '''

        edge = set(edge)
        if edge not in self.edges_by_idx:
            self.edges_by_idx[edge[0]].append(edge[1])
            self.edges_by_idx[edge[1]].append(edge[0])

    def _generate_edges(self):
        ''' 
        Helper function to generate set of edges within graph.

        Returns:
            (set) : set of edges within graph

        '''

        edges = []
        for vertex in self.edges_by_idx:
            for neighbour in self.edges_by_idx[vertex]:
                if {neighbour, vertex} not in edges:
                    edges.append((vertex, neighbour))
        
        return edges