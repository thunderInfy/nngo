package nngo

func (g *Graph) TopologicalSort(n *Node, visited Set[*Node], sorted *Stack[*Node], reverse bool) {
	visited[n] = true

	var neighbors [](*Node)
	if reverse {
		neighbors = n.Inputs
	} else {
		neighbors = n.Outputs
	}

	for i := range neighbors {
		if !visited[neighbors[i]] {
			g.TopologicalSort(neighbors[i], visited, sorted, reverse)
		}
	}

	sorted.Push(n)
}

func (g *Graph) Forward(inputValues []float64) (err error) {
	visited := Set[*Node]{}
	sorted := Stack[*Node]{}

	for _, inp := range g.Inputs {
		if !visited[inp] {
			g.TopologicalSort(inp, visited, &sorted, false)
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

func (g *Graph) Backprop(upstreamGrads []float64) {
	visited := Set[*Node]{}
	sorted := Stack[*Node]{}

	for _, out := range g.Outputs {
		if !visited[out] {
			g.TopologicalSort(out, visited, &sorted, true)
		}
	}

	for i := range g.Outputs {
		g.Outputs[i].Grad = upstreamGrads[i]
	}

	for {
		n, empty := sorted.Pop()
		if empty {
			break
		}
		n.ComputeGrad()
	}
}
