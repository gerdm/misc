# Implementation of the tree-generating stochastic process
# for a Bayesian CART model
# Refernces:
#   1. https://docs.juliaplots.org/latest/graphrecipes/examples/#AbstractTrees-Trees
#   2. https://github.com/JuliaCollections/AbstractTrees.jl/blob/master/examples/binarytree_core.jl

using Plots
using GraphRecipes
using AbstractTrees

d = Dict(:a => 2,:d => Dict(:b => 4,:c => "Hello"),:e => 5.0)
plot(TreePlot(d), method=:tree)

tree = [1:3, "foo", [[[4, 5], 6, 7], 8]]
print_tree(d)

mutable struct BinaryNode{T}
    data::T
    parent::BinaryNode{T}
    left::BinaryNode{T}
    right::BinaryNode{T}

    # Root constructor
    BinaryNode{T}(data) where T = new{T}(data)
    # Child node constructor
    BinaryNode{T}(data, parent::BinaryNode{T}) where T = new{T}(data, parent)
end

BinaryNode(data) = BinaryNode{typeof(data)}(data)

function leftchild(data, parent::BinaryNode)
    !isdefined(parent, :left) || error("left child is already assigned")
    node = typeof(parent)(data, parent)
    parent.left = node
end

function rightchild(data, parent::BinaryNode)
    !isdefined(parent, :right) || error("right child is already assigned")
    node = typeof(parent)(data, parent)
    parent.right = node
end