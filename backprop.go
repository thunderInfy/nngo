package nngo

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

func (g *Graph) TopologicalSortUtil(n *Node, visited Set[*Node], sorted *Stack[*Node]) {
	// visit node n
	visited[n] = true

	// visit its child - currently there is only atmost one child
	if n.Output != nil && !visited[n.Output] {
		g.TopologicalSortUtil(n.Output, visited, sorted)
	}

	// no more children left, so put it in the sorted stack
	sorted.Push(n)
}

// topological sort
func (g *Graph) Forward(inputValues []float64) (err error) {
	visited := Set[*Node]{}
	sorted := Stack[*Node]{}

	for _, inp := range g.Inputs {
		if !visited[inp] {
			g.TopologicalSortUtil(inp, visited, &sorted)
		}
	}

	err = g.SetInputs(inputValues)
	if err != nil {
		return
	}

	for {
		n, empty := sorted.Pop()
		if empty {
			break
		}
		n.ComputeVal()
	}
	return
}

func (g *Graph) Backprop(upstreamGrad float64) {
	g.Output.Backprop(upstreamGrad)
}
