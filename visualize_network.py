"""Visualize a snapshot of SNOMED graph

@author: Nikhil Pattisapu, iREL, IIIT-H"""


import re
import pickle
from graphviz import Digraph


def remove_paran(concept_name):
    """Strips text between brackets"""
    return re.sub("[\(\[].*?[\)\]]", "", concept_name)


def draw_graph():
    """Draws SNOMED graph"""
    snmd_graph = Digraph(engine='neato', format='eps')
    snmd_graph.attr('node', fontsize='10')
    snmd_graph.attr('edge', fontsize='3')
    snmd_graph.attr('node', height='0.08')
    snmd_graph.attr('node', width='0.1')
    snmd_graph.attr('edge', arrowsize='0.3')
    sid_desc = pickle.load(open('resources/sid_to_desc.pkl', 'rb'))
    edge_list = open('resources/edge_list.txt', 'r').readlines()
    accepted_vert = [233604007, 196112005, 11389007, 233619008]
    max_edges = 7
    edge_count = 0
    curr_edges = []

    for edge in edge_list:
        v0, v1 = edge.strip().split(' ')
        v0, v1 = int(v0), int(v1)
        if v0 in accepted_vert or v1 in accepted_vert:
            if edge_count < max_edges:
                v0, v1 = remove_paran(sid_desc[v0]), remove_paran(sid_desc[v1])
                if (v0, v1) not in curr_edges:
                    snmd_graph.edge(v0, v1, constraint='false')
                    curr_edges.append((v0, v1))
                    edge_count += 1

    snmd_graph.render('snomed_graph', view=True)

draw_graph()
