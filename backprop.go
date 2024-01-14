package nngo

import (
	"container/list"
	"errors"
)

// dfs currently, can be implemented as bfs too
func (n *Node) Backprop(upstreamGrad float64) {
	n.Grad = upstreamGrad
	switch n.Op {
	case Add:
		for _, inp := range n.Inputs {
			inp.Backprop(upstreamGrad)
		}
	case Multiply:
		n.Inputs[0].Backprop(upstreamGrad * n.Inputs[1].Val)
		n.Inputs[1].Backprop(upstreamGrad * n.Inputs[0].Val)
	case "":
		if n.IsOutputSymbol() {
			n.Inputs[0].Backprop(upstreamGrad)
		}
	}
}

// bfs
func (g *Graph) Forward(inputValues []float64) (err error) {
	err = g.SetInputs(inputValues)
	if err != nil {
		return
	}
	isNodePresent := make(map[*Node]bool)
	queue := list.New()
	for _, node := range g.Inputs {
		queue.PushBack(node)
		isNodePresent[node] = true
	}

	for front := queue.Front(); front != nil; front = queue.Front() {
		switch x := (front.Value).(type) {
		case *Node:
			x.ComputeVal()
			if x.Output != nil && !isNodePresent[x.Output] {
				queue.PushBack(x.Output)
			}
		default:
			err = errors.New("queue must contain node pointer")
			return
		}
		queue.Remove(front)
	}
	return
}

func (g *Graph) Backprop(upstreamGrad float64) {
	g.Output.Backprop(upstreamGrad)
}
