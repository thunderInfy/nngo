package nngo

func (g *Graph) TopologicalSortUtil(n *Node, visited Set[*Node], sorted *Stack[*Node], reverse bool) {
	visited[n] = true

	var neighbors [](*Node)
	if reverse {
		neighbors = n.Inputs
	} else {
		neighbors = n.Outputs
	}

	for _, neighbor := range neighbors {
		if !visited[neighbor] {
			g.TopologicalSortUtil(neighbor, visited, sorted, reverse)
		}
	}

	sorted.Push(n)
}

func (g *Graph) Forward(inputValues []float64) (err error) {
	visited := Set[*Node]{}
	sorted := Stack[*Node]{}

	for _, inp := range g.Inputs {
		if !visited[inp] {
			g.TopologicalSortUtil(inp, visited, &sorted, false)
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
	visited := Set[*Node]{}
	sorted := Stack[*Node]{}

	g.TopologicalSortUtil(g.Output, visited, &sorted, true)

	g.Output.Grad = upstreamGrad

	for {
		n, empty := sorted.Pop()
		if empty {
			break
		}
		n.ComputeGrad()
	}
}
