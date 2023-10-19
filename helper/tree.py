#!/usr/bin/env python
# coding: utf-8

import numpy as np


class Node:
    def __init__(self, label):
        """
        Node class to build label tree
        """
        self.label = label
        self.parent = None
        self.children = []

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

def lca_height(true_label, predicted_label):
    """
    Build the tree hierarchy-->{0: {1, 2}, 1: {3, 4, 5}, 2: {6, 7}, 6: {8, 9}, 7: {10, 11}}
    """
    root = Node(0)
    node_a = Node(1)
    node_b = Node(2)
    node_c = Node(3)
    node_d = Node(4)
    node_e = Node(5)
    node_f = Node(6)
    node_g = Node(7)
    node_h = Node(8)
    node_i = Node(9)
    node_j = Node(10)
    node_k = Node(11)
    root.add_child(node_a)
    root.add_child(node_b)
    node_a.add_child(node_c)
    node_a.add_child(node_d)
    node_a.add_child(node_e)
    node_b.add_child(node_f)
    node_b.add_child(node_g)
    node_f.add_child(node_h)
    node_f.add_child(node_i)
    node_g.add_child(node_j)
    node_g.add_child(node_k)

    # Find the LCA and calculate its height
    true_node = find_node(root, true_label)
    predicted_node = find_node(root, predicted_label)
    lca_node = find_lca(true_node, predicted_node)
    lca_height = get_height(lca_node)

    return lca_height

def find_node(node, label):
    if node.label == label:
        return node
    for child in node.children:
        found_node = find_node(child, label)
        if found_node is not None:
            return found_node
    return None

def find_lca(node1, node2):
    path1 = get_path_to_root(node1)
    path2 = get_path_to_root(node2)
    # i = 0
    # while i < min(len(path1), len(path2)) and path1[i] != path2[i]:
    #     i += 1
    # return path1[i]
    return [path for path in path1 if path in path2][0]

def get_path_to_root(node):
    path = [node]
    while node.parent is not None:
        path.append(node.parent)
        node = node.parent
    return path

def get_height(node):
    if len(node.children) == 0:
        return 0
    max_child_height = max([get_height(child) for child in node.children])
    return max_child_height + 1



if __name__ == "__main__":

    # code below is the use demo
    true_label = 3
    predicted_label = 4
    lca_heights = lca_height(true_label, predicted_label)
    print("LCA height:", lca_heights)