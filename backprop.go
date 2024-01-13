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
		n.Inputs[0].Backprop(upstreamGrad)
		n.Inputs[1].Backprop(upstreamGrad)
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
	isNodePresent := make(map[string]bool)
	queue := list.New()
	for _, node := range g.Inputs {
		queue.PushBack(node)
		isNodePresent[node.Label] = true
	}
	for {
		front := queue.Front()
		if front != nil {
			switch x := (front.Value).(type) {
			case *Node:
				x.ComputeVal()
				if x.Output != nil && !isNodePresent[x.Output.Label] {
					queue.PushBack(x.Output)
				}
			default:
				panic(errors.New("queue must contain node pointer"))
			}
			queue.Remove(front)
		} else {
			break
		}
	}
	return
}

func (g *Graph) Backprop(upstreamGrad float64) {
	g.Output.Backprop(upstreamGrad)
}
